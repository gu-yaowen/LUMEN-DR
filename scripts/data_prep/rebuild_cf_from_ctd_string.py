#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
CTD_ROOT = DATA_ROOT / "reference" / "ctd"
STRING_ROOT = DATA_ROOT / "reference" / "string"
REPORT_ROOT = ROOT / "reports"


def iter_ctd_rows(path: Path):
    header = None
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith("#"):
                if "\t" in line and line.startswith("# "):
                    maybe = line[2:].rstrip("\n").split("\t")
                    if maybe and maybe[0] != "The Comparative Toxicogenomics Database (CTD) - http://ctdbase.org/":
                        header = maybe
                continue
            if header is None:
                raise RuntimeError(f"Missing header in {path}")
            vals = line.rstrip("\n").split("\t")
            if len(vals) < len(header):
                vals += [""] * (len(header) - len(vals))
            yield dict(zip(header, vals))


def norm_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_ctd_chemical_aliases() -> tuple[dict[str, str], dict[str, str]]:
    alias_to_ctd = {}
    ctd_to_name = {}
    for row in iter_ctd_rows(CTD_ROOT / "CTD_chemicals.tsv.gz"):
        cid = row["ChemicalID"].strip()
        ctd_to_name[cid] = row.get("ChemicalName", "").strip()
        names = [row.get("ChemicalName", "")]
        names.extend(x.strip() for x in row.get("MESHSynonyms", "").split("|") if x.strip())
        names.extend(x.strip() for x in row.get("CTDCuratedSynonyms", "").split("|") if x.strip())
        for name in names:
            key = norm_text(name)
            if key:
                alias_to_ctd.setdefault(key, cid)
    return alias_to_ctd, ctd_to_name


def build_drug_mapping(drug_df: pd.DataFrame, alias_to_ctd: dict[str, str], ctd_to_name: dict[str, str]) -> tuple[dict[str, int], pd.DataFrame]:
    ctd_to_local = {}
    rows = []
    for row in drug_df.itertuples(index=False):
        drug_id = str(row.Drug)
        drug_name = str(getattr(row, "Name", ""))
        ctd_id = alias_to_ctd.get(norm_text(drug_name)) or alias_to_ctd.get(norm_text(drug_id)) or ""
        if ctd_id:
            ctd_to_local[ctd_id] = int(row.ID)
        rows.append(
            {
                "dataset_drug_id": drug_id,
                "dataset_drug_name": drug_name,
                "ctd_chemical_id": ctd_id,
                "ctd_chemical_name": ctd_to_name.get(ctd_id, "") if ctd_id else "",
                "matched_alias": drug_name if ctd_id else "",
            }
        )
    return ctd_to_local, pd.DataFrame(rows)


def load_gene_id_set() -> set[str]:
    out = set()
    for row in iter_ctd_rows(CTD_ROOT / "CTD_genes.tsv.gz"):
        gid = row.get("GeneID", "").strip()
        if gid:
            out.add(gid)
    return out


def collect_drug_protein_edges(local_drug_name_map: dict[str, int]) -> tuple[set[tuple[int, str]], set[str]]:
    edges = set()
    genes = set()
    for row in iter_ctd_rows(CTD_ROOT / "CTD_chem_gene_ixns.tsv.gz"):
        if row.get("OrganismID", "").strip() != "9606":
            continue
        chem_name = norm_text(row.get("ChemicalName", ""))
        gid = row.get("GeneID", "").strip()
        if chem_name in local_drug_name_map and gid:
            edges.add((local_drug_name_map[chem_name], gid))
            genes.add(gid)
    return edges, genes


def collect_protein_disease_edges(
    candidate_genes: set[str],
    disease_id_to_local: dict[str, int],
    direct_evidence_only: bool = True,
) -> set[tuple[str, int]]:
    target_disease_ids = {f"MESH:{x}": int(local_id) for x, local_id in disease_id_to_local.items()}
    edges = set()
    for row in iter_ctd_rows(CTD_ROOT / "CTD_genes_diseases.tsv.gz"):
        gid = row.get("GeneID", "").strip()
        did = row.get("DiseaseID", "").strip()
        if gid not in candidate_genes or did not in target_disease_ids:
            continue
        if direct_evidence_only and not row.get("DirectEvidence", "").strip():
            continue
        edges.add((gid, target_disease_ids[did]))
    return edges


def load_sequence_supported_protein_map(dataset: str) -> tuple[pd.DataFrame, dict[str, dict]]:
    protein_df = pd.read_csv(DATA_ROOT / dataset / "protein.csv", dtype=str).fillna("")
    keep = protein_df["Sequence"].astype(str).str.strip().ne("")
    protein_df = protein_df.loc[keep].copy().reset_index(drop=True)
    if "ID" in protein_df.columns:
        protein_df = protein_df.drop(columns=["ID"])
    protein_df.insert(0, "ID", range(len(protein_df)))
    by_gene = {
        str(rec["Protein"]): rec
        for rec in protein_df.to_dict(orient="records")
    }
    return protein_df, by_gene


def write_ppi_subset(protein_ids: list[str], out_csv: Path) -> int:
    wanted = set(protein_ids)
    up_to_string = defaultdict(set)
    with gzip.open(STRING_ROOT / "9606.protein.aliases.v12.0.txt.gz", "rt", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            alias = row["alias"].strip()
            if alias in wanted:
                up_to_string[alias].add(row["#string_protein_id"].strip())

    string_to_local = defaultdict(set)
    local_map = {pid: i for i, pid in enumerate(protein_ids)}
    for protein_id, string_ids in up_to_string.items():
        for sid in string_ids:
            string_to_local[sid].add(local_map[protein_id])

    ppi = set()
    with gzip.open(STRING_ROOT / "9606.protein.links.v12.0.txt.gz", "rt", encoding="utf-8", errors="ignore") as f:
        _ = f.readline()
        for line in f:
            p1, p2, score = line.strip().split()
            if p1 not in string_to_local or p2 not in string_to_local:
                continue
            for a in string_to_local[p1]:
                for b in string_to_local[p2]:
                    if a == b:
                        continue
                    x, y = sorted((a, b))
                    ppi.add((x, y, int(score)))

    pd.DataFrame(sorted(ppi), columns=["Protein1", "Protein2", "Score"]).to_csv(out_csv, index=False)
    return len(ppi)


def rebuild_dataset(dataset: str, direct_evidence_only: bool = True) -> dict:
    ds_dir = DATA_ROOT / dataset
    print(f"[{dataset}] load tables", flush=True)
    drug_df = pd.read_csv(ds_dir / "drug.csv", dtype=str).fillna("")
    disease_df = pd.read_csv(ds_dir / "disease.csv", dtype=str).fillna("")
    print(f"[{dataset}] load CTD aliases", flush=True)
    alias_to_ctd, ctd_to_name = load_ctd_chemical_aliases()
    ctd_to_local_drug, mapping_df = build_drug_mapping(drug_df, alias_to_ctd, ctd_to_name)
    local_drug_name_map = {norm_text(str(row.Name)): int(row.ID) for row in drug_df.itertuples(index=False)}

    print(f"[{dataset}] collect drug-protein", flush=True)
    dp_edges_raw, candidate_genes = collect_drug_protein_edges(local_drug_name_map)
    print(f"[{dataset}] collect protein-disease", flush=True)
    pd_edges_raw = collect_protein_disease_edges(
        candidate_genes=candidate_genes,
        disease_id_to_local={str(row.Disease): int(row.ID) for row in disease_df.itertuples(index=False)},
        direct_evidence_only=direct_evidence_only,
    )

    print(f"[{dataset}] load sequence-supported proteins", flush=True)
    valid_gene_ids = load_gene_id_set()
    candidate_protein_ids = sorted({gid for _, gid in dp_edges_raw} | {gid for gid, _ in pd_edges_raw})
    candidate_protein_ids = [gid for gid in candidate_protein_ids if gid in valid_gene_ids]

    existing_protein_df, existing_by_gene = load_sequence_supported_protein_map(dataset)
    kept_proteins = [gid for gid in candidate_protein_ids if gid in existing_by_gene]
    protein_out = pd.DataFrame([existing_by_gene[gid] for gid in kept_proteins]).reset_index(drop=True)
    protein_out["ID"] = range(len(protein_out))
    protein_out = protein_out[["ID"] + [c for c in protein_out.columns if c != "ID"]]
    gene_to_new = {gid: idx for idx, gid in enumerate(protein_out["Protein"].astype(str))}

    dp_edges = sorted((drug_id, gene_to_new[gid]) for drug_id, gid in dp_edges_raw if gid in gene_to_new)
    pd_edges = sorted((gene_to_new[gid], disease_id) for gid, disease_id in pd_edges_raw if gid in gene_to_new)

    drug_out = drug_df.copy()
    disease_out = disease_df.copy()

    drug_out.to_csv(ds_dir / "drug.csv", index=False)
    disease_out.to_csv(ds_dir / "disease.csv", index=False)
    protein_out.to_csv(ds_dir / "protein.csv", index=False)
    pd.DataFrame(dp_edges, columns=["Drug", "Protein"]).to_csv(ds_dir / "drug_protein.csv", index=False)
    pd.DataFrame(pd_edges, columns=["Protein", "Disease"]).to_csv(ds_dir / "protein_disease.csv", index=False)
    mapping_df.to_csv(ds_dir / "drug_ctd_mapping.csv", index=False)
    print(f"[{dataset}] build PPI subset", flush=True)
    ppi_n = write_ppi_subset(protein_out["Protein"].astype(str).tolist(), ds_dir / "protein_protein.csv")
    print(f"[{dataset}] done", flush=True)

    return {
        "dataset": dataset,
        "direct_evidence_only": direct_evidence_only,
        "mapped_drugs_to_ctd": int((mapping_df["ctd_chemical_id"].astype(str) != "").sum()),
        "candidate_genes_from_drug_protein": len(candidate_genes),
        "kept_proteins_with_sequence": len(protein_out),
        "n_drug_protein": len(dp_edges),
        "n_protein_disease": len(pd_edges),
        "n_protein_protein": ppi_n,
    }


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    reports = [rebuild_dataset("Cdataset", direct_evidence_only=True), rebuild_dataset("Fdataset", direct_evidence_only=True)]
    out = REPORT_ROOT / "cf_rebuild_from_ctd_string_report.json"
    out.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
