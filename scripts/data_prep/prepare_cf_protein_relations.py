#!/usr/bin/env python3
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
import re
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LLM_DATA = ROOT / "LLM_DDA" / "data"
CTD_RAW = ROOT / "ctd_raw"
STRING_RAW = ROOT / "string_raw"


def iter_ctd_rows(path: Path):
    header = None
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith("#"):
                if "\t" in line and line.startswith("# "):
                    maybe_header = line[2:].rstrip("\n").split("\t")
                    if maybe_header and maybe_header[0] not in {
                        "The Comparative Toxicogenomics Database (CTD) - http://ctdbase.org/"
                    }:
                        header = maybe_header
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


def load_ctd_chemical_aliases():
    alias_to_ctd = {}
    ctd_to_name = {}
    for row in iter_ctd_rows(CTD_RAW / "CTD_chemicals.tsv.gz"):
        cid = row["ChemicalID"]
        ctd_to_name[cid] = row.get("ChemicalName", "")
        names = [row.get("ChemicalName", "")]
        names.extend([x.strip() for x in row.get("MESHSynonyms", "").split("|") if x.strip()])
        names.extend(
            [x.strip() for x in row.get("CTDCuratedSynonyms", "").split("|") if x.strip()]
        )
        for name in names:
            key = norm_text(name)
            if key:
                alias_to_ctd.setdefault(key, cid)
    return alias_to_ctd, ctd_to_name


def load_gene_maps():
    gid_to_symbol = {}
    gid_to_symbol = {}
    for row in iter_ctd_rows(CTD_RAW / "CTD_genes.tsv.gz"):
        gid = row["GeneID"]
        gid_to_symbol[gid] = row.get("GeneSymbol", "")
    return gid_to_symbol


def build_for_dataset(ds_name: str, alias_to_ctd: dict, ctd_to_name: dict, gid_to_symbol: dict):
    ddir = LLM_DATA / ds_name
    drugs = pd.read_csv(ddir / "drug.csv")
    diseases = pd.read_csv(ddir / "disease.csv")

    dbid_to_local = dict(zip(drugs["Drug"], drugs["ID"]))
    diseaseid_to_local = dict(zip(diseases["Disease"], diseases["ID"]))
    dbid_to_ctd = {}
    mapping_rows = []
    for row in drugs.itertuples(index=False):
        candidates = [
            getattr(row, "Name", ""),
            getattr(row, "Drug", ""),
        ]
        ctd_id = None
        matched_alias = ""
        for cand in candidates:
            key = norm_text(str(cand))
            if key in alias_to_ctd:
                ctd_id = alias_to_ctd[key]
                matched_alias = cand
                break
        if ctd_id is not None:
            dbid_to_ctd[row.Drug] = ctd_id
        mapping_rows.append(
            {
                "dataset_drug_id": row.Drug,
                "dataset_drug_name": row.Name,
                "ctd_chemical_id": ctd_id or "",
                "ctd_chemical_name": ctd_to_name.get(ctd_id, "") if ctd_id else "",
                "matched_alias": matched_alias,
            }
        )

    matched_drug_names = {
        norm_text(str(row.Name)): row.ID for row in drugs.itertuples(index=False)
    }
    needed_dis_ids = {f"MESH:{x}" for x in diseaseid_to_local.keys()}

    gene_keys = set()
    dp_raw = set()
    for row in iter_ctd_rows(CTD_RAW / "CTD_chem_gene_ixns.tsv.gz"):
        if row.get("OrganismID", "") != "9606":
            continue
        gid = row.get("GeneID", "")
        chem_name_key = norm_text(row.get("ChemicalName", ""))
        local_drug = matched_drug_names.get(chem_name_key)
        if local_drug is not None and gid:
            gene_keys.add(gid)
            dp_raw.add((local_drug, gid))

    matched_disease_names = {
        norm_text(str(row.Name).split(";")[0]): row.ID for row in diseases.itertuples(index=False)
    }

    pd_raw = set()
    for row in iter_ctd_rows(CTD_RAW / "CTD_genes_diseases.tsv.gz"):
        gid = row.get("GeneID", "")
        dis_name_key = norm_text(row.get("DiseaseName", "").split(";")[0])
        local_dis = matched_disease_names.get(dis_name_key)
        if gid in gene_keys and local_dis is not None:
            pd_raw.add((gid, local_dis))

    protein_ids = sorted({gid for gid in gene_keys if gid in gid_to_symbol})
    protein_to_local = {p: i for i, p in enumerate(protein_ids)}

    drug_protein_edges = set()
    for local_drug, gid in dp_raw:
        if gid in protein_to_local:
            drug_protein_edges.add((local_drug, protein_to_local[gid]))

    protein_disease_edges = set()
    for gid, local_dis in pd_raw:
        if gid in protein_to_local:
            protein_disease_edges.add((protein_to_local[gid], local_dis))

    ddir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Protein": protein_ids}).to_csv(ddir / "protein.csv", index=False)
    pd.DataFrame(sorted(drug_protein_edges), columns=["Drug", "Protein"]).to_csv(
        ddir / "drug_protein.csv", index=False
    )
    pd.DataFrame(sorted(protein_disease_edges), columns=["Protein", "Disease"]).to_csv(
        ddir / "protein_disease.csv", index=False
    )
    pd.DataFrame(mapping_rows).to_csv(ddir / "drug_ctd_mapping.csv", index=False)

    return {
        "dataset": ds_name,
        "n_drugs": len(drugs),
        "n_diseases": len(diseases),
        "mapped_drugs_to_ctd": len(
            [x for x in dbid_to_local.keys() if x in dbid_to_ctd]
        ),
        "n_proteins": len(protein_ids),
        "n_drug_protein": len(drug_protein_edges),
        "n_protein_disease": len(protein_disease_edges),
    }, protein_ids


def build_string_ppi_subgraph(protein_ids, out_csv: Path):
    wanted = set(protein_ids)
    up_to_string = defaultdict(set)
    with gzip.open(
        STRING_RAW / "9606.protein.aliases.v12.0.txt.gz",
        "rt",
        encoding="utf-8",
        errors="ignore",
    ) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            alias = r["alias"].strip()
            if alias in wanted:
                up_to_string[alias].add(r["#string_protein_id"].strip())

    string_to_local = defaultdict(set)
    local_map = {p: i for i, p in enumerate(protein_ids)}
    for up, sids in up_to_string.items():
        for sid in sids:
            string_to_local[sid].add(local_map[up])

    ppi = set()
    with gzip.open(
        STRING_RAW / "9606.protein.links.v12.0.txt.gz",
        "rt",
        encoding="utf-8",
        errors="ignore",
    ) as f:
        header = f.readline()
        for line in f:
            p1, p2, score = line.strip().split()
            if int(score) < 700:
                continue
            if p1 not in string_to_local or p2 not in string_to_local:
                continue
            for a in string_to_local[p1]:
                for b in string_to_local[p2]:
                    if a == b:
                        continue
                    x, y = sorted((a, b))
                    ppi.add((x, y, int(score)))

    pd.DataFrame(sorted(ppi), columns=["Protein1", "Protein2", "Score"]).to_csv(
        out_csv, index=False
    )
    return len(ppi)


def main():
    alias_to_ctd, ctd_to_name = load_ctd_chemical_aliases()
    gid_to_symbol = load_gene_maps()

    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["Cdataset", "Fdataset"]
    reports = []
    for ds in datasets:
        rpt, proteins = build_for_dataset(ds, alias_to_ctd, ctd_to_name, gid_to_symbol)
        ppi_n = build_string_ppi_subgraph(
            proteins, LLM_DATA / ds / "protein_protein.csv"
        )
        rpt["n_protein_protein"] = ppi_n
        reports.append(rpt)

    with open(ROOT / "data_prep_report_cf.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
