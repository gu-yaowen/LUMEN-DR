#!/usr/bin/env python3
import csv
import gzip
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "Rdataset"
REPORT_ROOT = ROOT / "reports"
CTD_PATH = ROOT / "data" / "reference" / "ctd" / "CTD_genes_diseases.tsv.gz"


def load_valid_diseases():
    disease_df = pd.read_csv(DATA_ROOT / "disease.csv", dtype=str).fillna("")
    disease_ids = set(disease_df["Disease"].astype(str))
    disease_lookup = {str(row["Disease"]): str(row["ID"]) for _, row in disease_df.iterrows()}
    return disease_ids, disease_lookup


def load_protein_gene():
    protein_gene_df = pd.read_csv(DATA_ROOT / "protein_gene.csv", dtype=str).fillna("")
    gene_to_proteins = {}
    for gene, sub in protein_gene_df.groupby("Gene"):
        gene_to_proteins[str(gene)] = sorted(set(sub["Protein"].astype(str)))
    return protein_gene_df, gene_to_proteins


def build_protein_disease():
    disease_ids, _ = load_valid_diseases()
    protein_gene_df, gene_to_proteins = load_protein_gene()

    rows = []
    seen = set()
    matched_ctd_diseases = set()
    matched_genes = set()

    with gzip.open(CTD_PATH, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            gene_id = parts[1].strip()
            disease_id_raw = parts[3].strip()
            direct_evidence = parts[4].strip()
            if not gene_id or not disease_id_raw or not direct_evidence:
                continue
            if ":" in disease_id_raw:
                namespace, disease_id = disease_id_raw.split(":", 1)
                if namespace != "MESH":
                    continue
            else:
                disease_id = disease_id_raw
            if disease_id not in disease_ids:
                continue
            proteins = gene_to_proteins.get(gene_id)
            if not proteins:
                continue
            matched_ctd_diseases.add(disease_id)
            matched_genes.add(gene_id)
            for protein in proteins:
                key = (protein, disease_id)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"Protein": protein, "Disease": disease_id})

    out_df = pd.DataFrame(rows, columns=["Protein", "Disease"])
    out_df.to_csv(DATA_ROOT / "protein_disease.csv", index=False)

    return {
        "dataset": "Rdataset",
        "n_proteins": int(pd.read_csv(DATA_ROOT / "protein.csv", dtype=str).shape[0]),
        "n_diseases": int(pd.read_csv(DATA_ROOT / "disease.csv", dtype=str).shape[0]),
        "n_protein_gene_before_removal": int(protein_gene_df.shape[0]),
        "genes_with_protein_mapping": len(gene_to_proteins),
        "matched_ctd_genes": len(matched_genes),
        "matched_ctd_diseases": len(matched_ctd_diseases),
        "direct_evidence_only": True,
        "n_protein_disease": int(out_df.shape[0]),
    }


def remove_protein_gene():
    path = DATA_ROOT / "protein_gene.csv"
    if path.exists():
        path.unlink()


def refresh_statistics():
    stats_path = ROOT / "data" / "dataset_statistics.md"
    if not stats_path.exists():
        return
    # Leave markdown refresh to a separate step if needed.
    return


def main():
    if not CTD_PATH.exists():
        raise FileNotFoundError(CTD_PATH)
    summary = build_protein_disease()
    remove_protein_gene()
    out = REPORT_ROOT / "rdataset_protein_disease_from_ctd_report.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
