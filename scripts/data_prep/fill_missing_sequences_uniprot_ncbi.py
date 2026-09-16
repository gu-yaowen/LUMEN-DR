#!/usr/bin/env python3
import csv
import gzip
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "datasets"
CACHE_ROOT = ROOT / "data" / "reference"
CACHE_ROOT.mkdir(exist_ok=True)
NCBI_UNIPROT_MAP = ROOT / "data" / "datasets" / "result.csv"


def read_text_url(url: str, retries: int = 3, sleep_s: float = 2.0) -> str:
    last = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read().decode("utf-8")
        except Exception as e:
            last = e
            time.sleep(sleep_s)
    raise last


def parse_fasta(text: str):
    seqs = {}
    current = None
    chunks = []
    for line in text.splitlines():
        if not line:
            continue
        if line.startswith(">"):
            if current is not None:
                seqs[current] = "".join(chunks)
            current = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line.strip())
    if current is not None:
        seqs[current] = "".join(chunks)
    return seqs


def chunked(items, size):
    items = list(items)
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_uniprot_sequences(accessions):
    out = {}

    def fetch_one(acc):
        url = f"https://rest.uniprot.org/uniprotkb/{urllib.parse.quote(acc)}.fasta"
        try:
            text = read_text_url(url, retries=2, sleep_s=1.0)
        except Exception:
            return acc, ""
        fasta = parse_fasta(text)
        if not fasta:
            return acc, ""
        _, seq = next(iter(fasta.items()))
        return acc, seq

    ids = sorted(set(accessions))
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_one, acc): acc for acc in ids}
        for i, fut in enumerate(as_completed(futures), start=1):
            acc, seq = fut.result()
            if seq:
                out[acc] = seq
            if i % 100 == 0:
                print(f"[UniProt] fetched {i}/{len(ids)} accessions; hits={len(out)}", flush=True)
    return out


def ensure_gene2refseq():
    path = CACHE_ROOT / "gene2refseq.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_gene_to_uniprot_from_csv():
    if not NCBI_UNIPROT_MAP.exists():
        return {}
    df = pd.read_csv(NCBI_UNIPROT_MAP, dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    gene_col = "NCBI Gene"
    uniprot_col = "UniProt"
    mapping = defaultdict(list)
    for _, row in df.iterrows():
        gene = str(row.get(gene_col, "")).strip()
        acc = str(row.get(uniprot_col, "")).strip()
        if gene and acc:
            mapping[gene].append(acc)
    return {k: list(dict.fromkeys(v)) for k, v in mapping.items()}


def load_human_gene_to_refseq_protein(target_geneids):
    mapping = defaultdict(list)
    gene2refseq = ensure_gene2refseq()
    scanned = 0
    with gzip.open(gene2refseq, "rt", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            scanned += 1
            if scanned % 2_000_000 == 0:
                print(
                    f"[gene2refseq] scanned {scanned} rows; matched genes={len(mapping)}",
                    flush=True,
                )
            if row["#tax_id"] != "9606":
                continue
            geneid = row["GeneID"]
            if geneid not in target_geneids:
                continue
            prot = row["protein_accession.version"].strip()
            status = row.get("status", "").strip()
            if prot and prot != "-" and prot.startswith(("NP_", "XP_", "YP_", "WP_")):
                mapping[geneid].append((prot, status))
    return mapping


def choose_best_refseq(acc_status_pairs):
    if not acc_status_pairs:
        return ""
    acc_status_pairs = list(dict.fromkeys(acc_status_pairs))
    priority = {"REVIEWED": 0, "VALIDATED": 1, "MODEL": 2, "INFERRED": 3, "PREDICTED": 4, "PROVISIONAL": 5}
    acc_prefix_priority = {"NP_": 0, "XP_": 1, "YP_": 2, "WP_": 3}
    acc_status_pairs.sort(
        key=lambda x: (
            priority.get(x[1].upper(), 9),
            acc_prefix_priority.get(x[0][:3], 9),
            x[0],
        )
    )
    return acc_status_pairs[0][0]


def fetch_ncbi_protein_sequences(refseq_accessions):
    out = {}
    for chunk in chunked(sorted(set(refseq_accessions)), 200):
        ids = ",".join(chunk)
        url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
            + urllib.parse.urlencode(
                {"db": "protein", "id": ids, "rettype": "fasta", "retmode": "text"}
            )
        )
        text = read_text_url(url)
        fasta = parse_fasta(text)
        for header_id, seq in fasta.items():
            acc = header_id.split(".")[0]
            out[acc] = seq
        print(
            f"[NCBI] fetched chunk size={len(chunk)}; cumulative hits={len(out)}",
            flush=True,
        )
        time.sleep(0.34)
    return out


def fill_bdataset():
    path = DATA_ROOT / "Bdataset" / "protein.csv"
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = df["Sequence"].astype(str).str.strip().eq("")
    ids = df.loc[missing, "Protein"].astype(str).tolist()
    fetched = fetch_uniprot_sequences(ids)
    fill_n = 0
    for idx in df.index[missing]:
        acc = str(df.at[idx, "Protein"])
        seq = fetched.get(acc, "")
        if seq:
            df.at[idx, "Sequence"] = seq
            fill_n += 1
    df.to_csv(path, index=False)
    return {
        "dataset": "Bdataset",
        "missing_before": len(ids),
        "fetched_from_uniprot": len(fetched),
        "filled_now": fill_n,
        "missing_after": int(df["Sequence"].astype(str).str.strip().eq("").sum()),
    }


def fill_geneid_dataset(dataset):
    path = DATA_ROOT / dataset / "protein.csv"
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = df["Sequence"].astype(str).str.strip().eq("")
    geneids = set(df.loc[missing, "Protein"].astype(str).tolist())
    print(f"[{dataset}] missing genes before={len(geneids)}", flush=True)

    gene_to_uniprot = load_gene_to_uniprot_from_csv()
    mapped_uniprots = []
    for gid in geneids:
        mapped_uniprots.extend(gene_to_uniprot.get(gid, []))
    mapped_uniprots = list(dict.fromkeys(mapped_uniprots))
    print(
        f"[{dataset}] genes with CSV UniProt mapping={sum(1 for g in geneids if g in gene_to_uniprot)}; "
        f"unique UniProt accessions={len(mapped_uniprots)}",
        flush=True,
    )
    fetched_uniprot = fetch_uniprot_sequences(mapped_uniprots) if mapped_uniprots else {}

    gene_to_refseq = load_human_gene_to_refseq_protein(geneids)
    chosen = {gid: choose_best_refseq(pairs) for gid, pairs in gene_to_refseq.items()}
    chosen_nonempty = [x for x in chosen.values() if x]
    print(
        f"[{dataset}] genes with human RefSeq protein={len(chosen_nonempty)}",
        flush=True,
    )
    fetched = fetch_ncbi_protein_sequences(chosen_nonempty)
    fill_n = 0
    filled_by_uniprot = 0
    filled_by_refseq = 0
    refseq_col = []
    uniprot_col = []
    for idx in df.index:
        gid = str(df.at[idx, "Protein"])
        mapped_accs = gene_to_uniprot.get(gid, [])
        chosen_uniprot = ""
        for acc in mapped_accs:
            if fetched_uniprot.get(acc, ""):
                chosen_uniprot = acc
                break
        uniprot_col.append(chosen_uniprot)
        acc = chosen.get(gid, "")
        refseq_col.append(acc)
        if not missing.at[idx]:
            continue
        seq = ""
        if chosen_uniprot:
            seq = fetched_uniprot.get(chosen_uniprot, "")
            if seq:
                filled_by_uniprot += 1
        if not seq and acc:
            seq = fetched.get(acc.split(".")[0], "")
            if seq:
                filled_by_refseq += 1
        if seq:
            df.at[idx, "Sequence"] = seq
            fill_n += 1
    df["MappedUniProt"] = uniprot_col
    df["RefSeqProtein"] = refseq_col
    df.to_csv(path, index=False)
    return {
        "dataset": dataset,
        "missing_before": int(missing.sum()),
        "geneids_with_uniprot_csv": int(sum(1 for g in geneids if g in gene_to_uniprot)),
        "uniprot_accessions_fetched": len(fetched_uniprot),
        "geneids_with_refseq": int(sum(1 for x in chosen.values() if x)),
        "filled_by_uniprot": filled_by_uniprot,
        "filled_by_refseq": filled_by_refseq,
        "filled_now": fill_n,
        "missing_after": int(df["Sequence"].astype(str).str.strip().eq("").sum()),
    }


def main():
    targets = sys.argv[1:] or ["Bdataset", "Cdataset", "Fdataset"]
    reports = []
    for target in targets:
        if target == "Bdataset":
            reports.append(fill_bdataset())
        elif target in {"Cdataset", "Fdataset"}:
            reports.append(fill_geneid_dataset(target))
        else:
            raise ValueError(f"Unsupported dataset: {target}")
    out = ROOT / "reports" / "sequence_fill_report_uniprot_ncbi.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
