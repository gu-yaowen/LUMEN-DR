#!/usr/bin/env python3
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "LLM_DDA" / "data"
STRING_RAW = ROOT / "string_raw"


def load_string_sequences():
    seqs = {}
    current_id = None
    chunks = []
    with gzip.open(
        STRING_RAW / "9606.protein.sequences.v12.0.fa.gz",
        "rt",
        encoding="utf-8",
        errors="ignore",
    ) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    seqs[current_id] = "".join(chunks)
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if current_id is not None:
        seqs[current_id] = "".join(chunks)
    return seqs


def load_alias_map():
    alias_to_string = defaultdict(set)
    with gzip.open(
        STRING_RAW / "9606.protein.aliases.v12.0.txt.gz",
        "rt",
        encoding="utf-8",
        errors="ignore",
    ) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            alias = row["alias"].strip()
            sid = row["#string_protein_id"].strip()
            if alias and sid:
                alias_to_string[alias].add(sid)
    return alias_to_string


def choose_best_string_id(candidates, seqs):
    valid = [sid for sid in candidates if sid in seqs and seqs[sid]]
    if not valid:
        return "", ""
    best = max(valid, key=lambda sid: (len(seqs[sid]), sid))
    return best, seqs[best]


def fill_dataset(dataset, alias_to_string, seqs):
    protein_path = DATA_ROOT / dataset / "protein.csv"
    if not protein_path.exists():
        return {
            "dataset": dataset,
            "protein_file_exists": False,
            "protein_rows": 0,
            "sequence_filled": 0,
        }

    df = pd.read_csv(protein_path, dtype=str).fillna("")
    if "Protein" not in df.columns:
        raise RuntimeError(f"{protein_path} missing Protein column")

    string_ids = []
    sequences = []
    candidate_counts = []
    for protein_id in df["Protein"].astype(str):
        candidates = alias_to_string.get(protein_id, set())
        sid, seq = choose_best_string_id(candidates, seqs)
        string_ids.append(sid)
        sequences.append(seq)
        candidate_counts.append(len(candidates))

    df["StringProtein"] = string_ids
    df["Sequence"] = sequences
    df["StringAliasCandidateCount"] = candidate_counts
    df.to_csv(protein_path, index=False)

    return {
        "dataset": dataset,
        "protein_file_exists": True,
        "protein_rows": int(len(df)),
        "sequence_filled": int(sum(1 for x in sequences if x)),
        "sequence_missing": int(sum(1 for x in sequences if not x)),
    }


def main():
    seqs = load_string_sequences()
    alias_to_string = load_alias_map()
    reports = []
    for ds in ["Bdataset", "Cdataset", "Fdataset", "Rdataset"]:
        reports.append(fill_dataset(ds, alias_to_string, seqs))

    out = ROOT / "protein_sequence_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
