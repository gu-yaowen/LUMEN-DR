#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
REPORT_ROOT = ROOT / "reports"


def prune_dataset(dataset: str):
    ds_dir = DATA_ROOT / dataset
    protein_path = ds_dir / "protein.csv"
    drug_protein_path = ds_dir / "drug_protein.csv"
    protein_disease_path = ds_dir / "protein_disease.csv"
    protein_protein_path = ds_dir / "protein_protein.csv"

    protein_df = pd.read_csv(protein_path, dtype=str).fillna("")
    missing_mask = protein_df["Sequence"].astype(str).str.strip().eq("")
    removed_proteins = set(protein_df.loc[missing_mask, "Protein"].astype(str))
    kept_protein_df = protein_df.loc[~missing_mask].copy()

    drug_protein_df = pd.read_csv(drug_protein_path, dtype=str).fillna("")
    protein_disease_df = pd.read_csv(protein_disease_path, dtype=str).fillna("")
    protein_protein_df = pd.read_csv(protein_protein_path, dtype=str).fillna("")

    kept_drug_protein_df = drug_protein_df.loc[
        ~drug_protein_df["Protein"].astype(str).isin(removed_proteins)
    ].copy()
    kept_protein_disease_df = protein_disease_df.loc[
        ~protein_disease_df["Protein"].astype(str).isin(removed_proteins)
    ].copy()
    kept_protein_protein_df = protein_protein_df.loc[
        ~protein_protein_df["Protein1"].astype(str).isin(removed_proteins)
        & ~protein_protein_df["Protein2"].astype(str).isin(removed_proteins)
    ].copy()

    kept_protein_df.to_csv(protein_path, index=False)
    kept_drug_protein_df.to_csv(drug_protein_path, index=False)
    kept_protein_disease_df.to_csv(protein_disease_path, index=False)
    kept_protein_protein_df.to_csv(protein_protein_path, index=False)

    missing_list_path = REPORT_ROOT / f"{dataset}_missing_sequence_protein_ids.txt"
    missing_list_path.write_text("", encoding="utf-8")

    return {
        "dataset": dataset,
        "removed_proteins": len(removed_proteins),
        "remaining_proteins": len(kept_protein_df),
        "removed_drug_protein_edges": len(drug_protein_df) - len(kept_drug_protein_df),
        "remaining_drug_protein_edges": len(kept_drug_protein_df),
        "removed_protein_disease_edges": len(protein_disease_df) - len(kept_protein_disease_df),
        "remaining_protein_disease_edges": len(kept_protein_disease_df),
        "removed_protein_protein_edges": len(protein_protein_df) - len(kept_protein_protein_df),
        "remaining_protein_protein_edges": len(kept_protein_protein_df),
    }


def refresh_protein_sequence_report():
    items = []
    for dataset in ["Bdataset", "Cdataset", "Fdataset", "Rdataset"]:
        protein_path = DATA_ROOT / dataset / "protein.csv"
        if not protein_path.exists():
            items.append(
                {
                    "dataset": dataset,
                    "protein_file_exists": False,
                    "protein_rows": 0,
                    "sequence_filled": 0,
                    "sequence_missing": 0,
                }
            )
            continue
        df = pd.read_csv(protein_path, dtype=str).fillna("")
        missing = int(df["Sequence"].astype(str).str.strip().eq("").sum())
        items.append(
            {
                "dataset": dataset,
                "protein_file_exists": True,
                "protein_rows": len(df),
                "sequence_filled": len(df) - missing,
                "sequence_missing": missing,
            }
        )
    out = REPORT_ROOT / "protein_sequence_report.json"
    out.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


def refresh_cf_report():
    report_path = REPORT_ROOT / "data_prep_report_cf.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            items = json.load(f)
    else:
        items = []

    by_dataset = {item["dataset"]: item for item in items}
    for dataset in ["Cdataset", "Fdataset"]:
        ds_dir = DATA_ROOT / dataset
        protein_df = pd.read_csv(ds_dir / "protein.csv", dtype=str).fillna("")
        drug_protein_df = pd.read_csv(ds_dir / "drug_protein.csv", dtype=str).fillna("")
        protein_disease_df = pd.read_csv(ds_dir / "protein_disease.csv", dtype=str).fillna("")
        protein_protein_df = pd.read_csv(ds_dir / "protein_protein.csv", dtype=str).fillna("")
        entry = by_dataset.get(dataset, {"dataset": dataset})
        entry["n_proteins"] = len(protein_df)
        entry["n_drug_protein"] = len(drug_protein_df)
        entry["n_protein_disease"] = len(protein_disease_df)
        entry["n_protein_protein"] = len(protein_protein_df)
        by_dataset[dataset] = entry

    ordered = []
    for dataset in ["Cdataset", "Fdataset"]:
        if dataset in by_dataset:
            ordered.append(by_dataset[dataset])
    report_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    summary = [prune_dataset("Cdataset"), prune_dataset("Fdataset")]
    refresh_protein_sequence_report()
    refresh_cf_report()
    out = REPORT_ROOT / "remove_missing_sequence_proteins_report.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
