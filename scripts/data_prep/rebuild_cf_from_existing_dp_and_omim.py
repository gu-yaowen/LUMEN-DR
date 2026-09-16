#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
FEATURE_ROOT = ROOT / "features"
REPORT_ROOT = ROOT / "reports"
STRING_ROOT = DATA_ROOT / "reference" / "string"
CTD_ROOT = DATA_ROOT / "reference" / "ctd"


UNIPROT_RE = re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-Z0-9]{10}$")


def load_string_uniprot_aliases() -> dict[str, list[str]]:
    aliases: dict[str, set[str]] = defaultdict(set)
    with gzip.open(STRING_ROOT / "9606.protein.aliases.v12.0.txt.gz", "rt", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            string_id = row["#string_protein_id"].strip()
            alias = row["alias"].strip()
            source = row["source"].strip()
            if "UniProt" not in source:
                continue
            if not UNIPROT_RE.match(alias):
                continue
            aliases[string_id].add(alias)
    return {k: sorted(v) for k, v in aliases.items()}


def choose_uniprot(existing: str, candidates: list[str]) -> str:
    existing = existing.strip()
    if existing and UNIPROT_RE.match(existing):
        return existing
    return candidates[0] if candidates else ""


def load_existing_protein_frame(dataset: str) -> pd.DataFrame:
    protein_df = pd.read_csv(DATA_ROOT / dataset / "protein.csv", dtype=str).fillna("")
    protein_df["ID"] = protein_df["ID"].astype(int)
    protein_df["Protein"] = protein_df["Protein"].astype(str)
    return protein_df


def decode_drug_protein_to_gene_ids(dataset: str, protein_df: pd.DataFrame) -> set[tuple[int, str]]:
    protein_idx_to_gene = dict(zip(protein_df["ID"].astype(int), protein_df["Protein"].astype(str)))
    dp_df = pd.read_csv(DATA_ROOT / dataset / "drug_protein.csv", dtype=str).fillna("")
    edges: set[tuple[int, str]] = set()
    for row in dp_df.itertuples(index=False):
        drug_idx = int(row.Drug)
        protein_idx = int(row.Protein)
        gene_id = protein_idx_to_gene.get(protein_idx)
        if gene_id:
            edges.add((drug_idx, gene_id))
    return edges


def load_omim_gene_disease_edges(dataset: str) -> set[tuple[str, int]]:
    disease_df = pd.read_csv(DATA_ROOT / dataset / "disease.csv", dtype=str).fillna("")
    omim_to_local = {
        f"OMIM:{str(row.Disease).strip()[1:]}": int(row.ID)
        for row in disease_df.itertuples(index=False)
        if str(row.Disease).strip().startswith("D")
    }
    src = CTD_ROOT / f"CTD_genes_diseases_omim_{dataset}.csv"
    gd_df = pd.read_csv(
        src,
        dtype=str,
        names=[
            "GeneSymbol",
            "GeneID",
            "DiseaseName",
            "DiseaseID",
            "DirectEvidence",
            "InferenceChemicalName",
            "InferenceScore",
            "OmimIDs",
            "PubMedIDs",
        ],
        header=None,
        skiprows=1,
    ).fillna("")
    edges: set[tuple[str, int]] = set()
    for row in gd_df.itertuples(index=False):
        gene_id = str(row.GeneID).strip()
        disease_id = str(row.DiseaseID).strip()
        local_disease = omim_to_local.get(disease_id)
        if gene_id and local_disease is not None:
            edges.add((gene_id, local_disease))
    return edges


def build_final_proteins(dataset: str, protein_df: pd.DataFrame, needed_gene_ids: set[str], string_to_uniprot: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, int], dict[str, str]]:
    protein_df = protein_df.copy()
    protein_df["Sequence"] = protein_df["Sequence"].astype(str).str.strip()
    protein_df["StringProtein"] = protein_df["StringProtein"].astype(str).str.strip()
    if "MappedUniProt" not in protein_df.columns:
        protein_df["MappedUniProt"] = ""
    protein_df["MappedUniProt"] = protein_df["MappedUniProt"].astype(str).str.strip()
    protein_df = protein_df[protein_df["Protein"].astype(str).isin(needed_gene_ids)].copy()
    protein_df = protein_df[protein_df["Sequence"] != ""].copy()
    protein_df = protein_df[protein_df["StringProtein"] != ""].copy()

    protein_df["FinalUniProt"] = protein_df.apply(
        lambda row: choose_uniprot(row["MappedUniProt"], string_to_uniprot.get(row["StringProtein"], [])),
        axis=1,
    )
    protein_df = protein_df[protein_df["FinalUniProt"] != ""].copy()
    protein_df = protein_df.drop_duplicates(subset=["FinalUniProt"], keep="first").copy()
    protein_df = protein_df.sort_values(by=["FinalUniProt"]).reset_index(drop=True)
    if "ID" in protein_df.columns:
        protein_df = protein_df.drop(columns=["ID"])
    protein_df.insert(0, "ID", range(len(protein_df)))

    final_df = pd.DataFrame(
        {
            "ID": protein_df["ID"].astype(int),
            "Protein": protein_df["FinalUniProt"].astype(str),
            "GeneID": protein_df["Protein"].astype(str),
            "StringProtein": protein_df["StringProtein"].astype(str),
            "Sequence": protein_df["Sequence"].astype(str),
        }
    )
    gene_to_local = dict(zip(final_df["GeneID"].astype(str), final_df["ID"].astype(int)))
    gene_to_uniprot = dict(zip(final_df["GeneID"].astype(str), final_df["Protein"].astype(str)))
    return final_df, gene_to_local, gene_to_uniprot


def write_ppi_subset(final_protein_df: pd.DataFrame, out_csv: Path) -> int:
    string_to_local: dict[str, int] = dict(
        zip(final_protein_df["StringProtein"].astype(str), final_protein_df["ID"].astype(int))
    )
    edges: set[tuple[int, int, int]] = set()
    with gzip.open(STRING_ROOT / "9606.protein.links.v12.0.txt.gz", "rt", encoding="utf-8", errors="ignore") as handle:
        _ = handle.readline()
        for line in handle:
            p1, p2, score = line.strip().split()
            i = string_to_local.get(p1)
            j = string_to_local.get(p2)
            if i is None or j is None or i == j:
                continue
            a, b = sorted((i, j))
            edges.add((a, b, int(score)))
    pd.DataFrame(sorted(edges), columns=["Protein1", "Protein2", "Score"]).to_csv(out_csv, index=False)
    return len(edges)


def rewrite_protein_features(dataset: str, old_protein_df: pd.DataFrame, final_protein_df: pd.DataFrame) -> dict[str, int]:
    feat_path = FEATURE_ROOT / dataset / "ESMC_protein_emb.pkl"
    with feat_path.open("rb") as handle:
        feat = pickle.load(handle)

    old_gene_to_vector = {}
    for row in old_protein_df.itertuples(index=False):
        key = str(row.Protein)
        if key in feat:
            old_gene_to_vector[key] = feat[key]

    new_feat = {}
    missing = 0
    for row in final_protein_df.itertuples(index=False):
        gene_id = str(row.GeneID)
        protein_id = str(row.Protein)
        vec = old_gene_to_vector.get(gene_id)
        if vec is None:
            missing += 1
            continue
        new_feat[protein_id] = vec

    with feat_path.open("wb") as handle:
        pickle.dump(new_feat, handle)

    return {"written": len(new_feat), "missing_old_embedding": missing}


def rebuild_dataset(dataset: str, string_to_uniprot: dict[str, list[str]]) -> dict[str, int]:
    ds_dir = DATA_ROOT / dataset
    old_protein_df = load_existing_protein_frame(dataset)
    drug_protein_gene_edges = decode_drug_protein_to_gene_ids(dataset, old_protein_df)
    gene_disease_edges = load_omim_gene_disease_edges(dataset)

    needed_gene_ids = {gid for _, gid in drug_protein_gene_edges} | {gid for gid, _ in gene_disease_edges}
    final_protein_df, gene_to_local, gene_to_uniprot = build_final_proteins(
        dataset, old_protein_df, needed_gene_ids, string_to_uniprot
    )

    final_gene_set = set(gene_to_local)
    final_dp = sorted((drug_idx, gene_to_local[gene_id]) for drug_idx, gene_id in drug_protein_gene_edges if gene_id in final_gene_set)
    final_gd = sorted((gene_to_local[gene_id], disease_idx) for gene_id, disease_idx in gene_disease_edges if gene_id in final_gene_set)

    final_protein_df.to_csv(ds_dir / "protein.csv", index=False)
    pd.DataFrame(final_dp, columns=["Drug", "Protein"]).to_csv(ds_dir / "drug_protein.csv", index=False)
    pd.DataFrame(final_gd, columns=["Protein", "Disease"]).to_csv(ds_dir / "protein_disease.csv", index=False)
    ppi_n = write_ppi_subset(final_protein_df, ds_dir / "protein_protein.csv")
    feat_report = rewrite_protein_features(dataset, old_protein_df, final_protein_df)

    return {
        "n_input_drug_protein_gene_edges": len(drug_protein_gene_edges),
        "n_input_gene_disease_edges": len(gene_disease_edges),
        "n_final_proteins": len(final_protein_df),
        "n_final_drug_protein": len(final_dp),
        "n_final_protein_disease": len(final_gd),
        "n_final_protein_protein": ppi_n,
        "n_unique_uniprot": final_protein_df["Protein"].nunique(),
        "n_unique_geneid": final_protein_df["GeneID"].nunique(),
        "protein_feature_written": feat_report["written"],
        "protein_feature_missing_old_embedding": feat_report["missing_old_embedding"],
    }


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    string_to_uniprot = load_string_uniprot_aliases()
    report = {
        dataset: rebuild_dataset(dataset, string_to_uniprot)
        for dataset in ("Cdataset", "Fdataset")
    }
    out = REPORT_ROOT / "cf_rebuild_from_existing_dp_and_omim_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
