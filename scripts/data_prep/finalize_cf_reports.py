#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd

from prepare_cf_protein_relations import LLM_DATA, ROOT, build_string_ppi_subgraph


def collect(ds: str):
    ddir = LLM_DATA / ds
    proteins = pd.read_csv(ddir / "protein.csv")["Protein"].astype(str).tolist()
    ppi_n = build_string_ppi_subgraph(proteins, ddir / "protein_protein.csv")
    report = {
        "dataset": ds,
        "n_drugs": len(pd.read_csv(ddir / "drug.csv")),
        "n_diseases": len(pd.read_csv(ddir / "disease.csv")),
        "mapped_drugs_to_ctd": int(pd.read_csv(ddir / "drug_ctd_mapping.csv")["ctd_chemical_id"].fillna("").astype(str).ne("").sum()),
        "n_proteins": len(proteins),
        "n_drug_protein": len(pd.read_csv(ddir / "drug_protein.csv")),
        "n_protein_disease": len(pd.read_csv(ddir / "protein_disease.csv")),
        "n_protein_protein": ppi_n,
    }
    return report


def main():
    reports = [collect("Cdataset"), collect("Fdataset")]
    with open(ROOT / "data_prep_report_cf.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
