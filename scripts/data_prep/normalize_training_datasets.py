#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import os
import pickle
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
FEATURES_ROOT = ROOT / "features"
REPORT_ROOT = ROOT / "reports"
CTD_ROOT = DATA_ROOT / "reference" / "ctd"
REDDA_B_ROOT = Path(
    os.environ.get(
        "LUMEN_DR_REDDA_ROOT",
        ROOT / "external" / "REDDA" / "dataset" / "Bdataset",
    )
)


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


def load_embedding_ids(dataset: str, filename: str) -> set[str]:
    path = FEATURES_ROOT / dataset / filename
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, got {type(obj)}")
    return {str(k) for k in obj.keys()}


def add_or_reset_id_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    df.insert(0, "ID", range(len(df)))
    return df


def validate_dense_matrix(path: Path, n_drug: int, n_disease: int) -> list[list[int]]:
    mat = pd.read_csv(path, header=None)
    if mat.shape != (n_drug, n_disease):
        raise ValueError(f"{path} shape {mat.shape} != {(n_drug, n_disease)}")
    return mat


def validate_index_range(series: pd.Series, upper: int, name: str) -> None:
    vals = pd.to_numeric(series, errors="raise").astype(int)
    if len(vals) == 0:
        return
    if vals.min() < 0 or vals.max() >= upper:
        raise ValueError(f"{name} out of range: min={vals.min()} max={vals.max()} upper={upper}")


def validate_feature_coverage(dataset: str, drug_df: pd.DataFrame, disease_df: pd.DataFrame, protein_df: pd.DataFrame) -> dict:
    drug_ids = load_embedding_ids(dataset, "MolFormer_drug_emb.pkl")
    disease_ids = load_embedding_ids(dataset, "BERT_disease_emb.pkl")
    protein_ids = load_embedding_ids(dataset, "ESMC_protein_emb.pkl")

    missing_drug = sorted(set(drug_df["Drug"].astype(str)) - drug_ids)
    missing_disease = sorted(set(disease_df["Disease"].astype(str)) - disease_ids)
    missing_protein = sorted(set(protein_df["Protein"].astype(str)) - protein_ids)
    if missing_drug or missing_disease or missing_protein:
        raise ValueError(
            f"{dataset} missing features: "
            f"drug={len(missing_drug)} disease={len(missing_disease)} protein={len(missing_protein)}"
        )
    return {
        "drug_feature_count": len(drug_ids),
        "disease_feature_count": len(disease_ids),
        "protein_feature_count": len(protein_ids),
    }


def load_ctd_alias_to_chemical() -> dict[str, str]:
    out: dict[str, str] = {}
    for row in iter_ctd_rows(CTD_ROOT / "CTD_chemicals.tsv.gz"):
        cid = row["ChemicalID"]
        names = [row.get("ChemicalName", "")]
        names.extend([x.strip() for x in row.get("MESHSynonyms", "").split("|") if x.strip()])
        names.extend([x.strip() for x in row.get("CTDCuratedSynonyms", "").split("|") if x.strip()])
        for name in names:
            key = norm_text(name)
            if key:
                out.setdefault(key, cid)
    return out


def load_ctd_gene_ids() -> set[str]:
    out = set()
    for row in iter_ctd_rows(CTD_ROOT / "CTD_genes.tsv.gz"):
        gid = row.get("GeneID", "").strip()
        if gid:
            out.add(gid)
    return out


def rebuild_cf_original_protein_ids(dataset: str) -> list[str]:
    ds_dir = DATA_ROOT / dataset
    drugs = pd.read_csv(ds_dir / "drug.csv", dtype=str).fillna("")
    diseases = pd.read_csv(ds_dir / "disease.csv", dtype=str).fillna("")

    alias_to_ctd = load_ctd_alias_to_chemical()
    valid_gene_ids = load_ctd_gene_ids()

    matched_drug_names = {norm_text(str(row.Name)): str(row.ID) for row in drugs.itertuples(index=False)}
    gene_keys = set()
    for row in iter_ctd_rows(CTD_ROOT / "CTD_chem_gene_ixns.tsv.gz"):
        if row.get("OrganismID", "") != "9606":
            continue
        gid = row.get("GeneID", "").strip()
        if not gid:
            continue
        chem_name_key = norm_text(row.get("ChemicalName", ""))
        if chem_name_key in matched_drug_names:
            gene_keys.add(gid)

    # Keep the same disease-name matching convention used in the original preparation script.
    matched_disease_names = {
        norm_text(str(row.Name).split(";")[0]): str(row.ID) for row in diseases.itertuples(index=False)
    }
    protein_ids = sorted(gid for gid in gene_keys if gid in valid_gene_ids)

    # Force the same traversal dependency as the original build: disease filtering only affects edges,
    # but we iterate once here so a missing CTD file fails early and explicitly.
    seen_any = False
    for row in iter_ctd_rows(CTD_ROOT / "CTD_genes_diseases.tsv.gz"):
        gid = row.get("GeneID", "").strip()
        dis_name_key = norm_text(row.get("DiseaseName", "").split(";")[0])
        if gid in gene_keys and dis_name_key in matched_disease_names:
            seen_any = True
            break
    if not seen_any:
        raise RuntimeError(f"{dataset}: failed to recover CTD overlap while rebuilding protein index space")
    return protein_ids


def remap_edge_file_via_old_protein_index(
    df: pd.DataFrame,
    protein_columns: list[str],
    old_protein_ids: list[str],
    retained_protein_to_new_idx: dict[str, int],
) -> tuple[pd.DataFrame, dict[str, int]]:
    old_idx_to_protein = pd.Series(old_protein_ids, index=range(len(old_protein_ids)))
    work = df.copy()
    reasons = {
        "source_rows": int(len(work)),
        "dropped_missing_retained_protein": 0,
        "kept_rows": 0,
    }
    keep_mask = pd.Series(True, index=work.index)
    for col in protein_columns:
        vals = pd.to_numeric(work[col], errors="raise").astype(int)
        if len(vals) and (vals.min() < 0 or vals.max() >= len(old_protein_ids)):
            raise ValueError(
                f"{col} old protein index out of range: min={vals.min()} max={vals.max()} "
                f"limit={len(old_protein_ids)}"
            )
        prot_ids = vals.map(old_idx_to_protein)
        keep_mask &= prot_ids.isin(retained_protein_to_new_idx)
        work[col] = prot_ids
    dropped = int((~keep_mask).sum())
    reasons["dropped_missing_retained_protein"] = dropped
    work = work.loc[keep_mask].copy()
    for col in protein_columns:
        work[col] = work[col].map(retained_protein_to_new_idx).astype(int)
    reasons["kept_rows"] = int(len(work))
    return work.reset_index(drop=True), reasons


def normalize_bdataset() -> dict:
    ds = "Bdataset"
    ds_dir = DATA_ROOT / ds
    redda_drug = pd.read_csv(REDDA_B_ROOT / "Omics" / "drug.csv", dtype=str).fillna("")
    redda_disease = pd.read_csv(REDDA_B_ROOT / "Omics" / "disease.csv", dtype=str).fillna("")
    redda_protein = pd.read_csv(REDDA_B_ROOT / "Omics" / "protein.csv", dtype=str).fillna("")

    drug_df = pd.read_csv(ds_dir / "drug.csv", dtype=str).fillna("")
    disease_df = pd.read_csv(ds_dir / "disease.csv", dtype=str).fillna("")
    protein_df = pd.read_csv(ds_dir / "protein.csv", dtype=str).fillna("")

    if drug_df["Drug"].astype(str).tolist() != redda_drug["Drug"].astype(str).tolist():
        raise ValueError("Bdataset drug order mismatch between local data and REDDA source")
    if disease_df["Disease"].astype(str).tolist() != redda_disease["Disease"].astype(str).tolist():
        raise ValueError("Bdataset disease order mismatch between local data and REDDA source")
    if protein_df["Protein"].astype(str).tolist() != redda_protein["Protein"].astype(str).tolist():
        raise ValueError("Bdataset protein order mismatch between local data and REDDA source")

    protein_feature_ids = load_embedding_ids(ds, "ESMC_protein_emb.pkl")
    keep_mask = protein_df["Sequence"].astype(str).str.strip().ne("") & protein_df["Protein"].astype(str).isin(protein_feature_ids)
    kept_protein_df = add_or_reset_id_column(protein_df.loc[keep_mask].reset_index(drop=True))
    retained_protein_ids = kept_protein_df["Protein"].astype(str).tolist()
    retained_map = {pid: idx for idx, pid in enumerate(retained_protein_ids)}

    dp_src = pd.read_csv(REDDA_B_ROOT / "Associations" / "drug_protein.csv", dtype=str).fillna("")
    pd_src = pd.read_csv(REDDA_B_ROOT / "Associations" / "protein_disease.csv", dtype=str).fillna("")
    ppi_src = pd.read_csv(REDDA_B_ROOT / "Interactions" / "protein_protein.csv", dtype=str).fillna("")
    ppi_src = ppi_src.rename(columns={"Combined Score": "Score"})

    remapped_dp, dp_stats = remap_edge_file_via_old_protein_index(
        dp_src, ["Protein"], redda_protein["Protein"].astype(str).tolist(), retained_map
    )
    remapped_pd, pd_stats = remap_edge_file_via_old_protein_index(
        pd_src, ["Protein"], redda_protein["Protein"].astype(str).tolist(), retained_map
    )
    remapped_ppi, ppi_stats = remap_edge_file_via_old_protein_index(
        ppi_src, ["Protein1", "Protein2"], redda_protein["Protein"].astype(str).tolist(), retained_map
    )

    drug_df = add_or_reset_id_column(drug_df)
    disease_df = add_or_reset_id_column(disease_df)
    validate_dense_matrix(ds_dir / "drug_dis.csv", len(drug_df), len(disease_df))

    drug_df.to_csv(ds_dir / "drug.csv", index=False)
    disease_df.to_csv(ds_dir / "disease.csv", index=False)
    kept_protein_df.to_csv(ds_dir / "protein.csv", index=False)
    remapped_dp.to_csv(ds_dir / "drug_protein.csv", index=False)
    remapped_pd.to_csv(ds_dir / "protein_disease.csv", index=False)
    remapped_ppi.to_csv(ds_dir / "protein_protein.csv", index=False)

    feature_stats = validate_feature_coverage(ds, drug_df, disease_df, kept_protein_df)
    return {
        "dataset": ds,
        "n_drug": int(len(drug_df)),
        "n_disease": int(len(disease_df)),
        "n_protein": int(len(kept_protein_df)),
        "protein_removed_missing_sequence_or_embedding": int((~keep_mask).sum()),
        "edge_stats": {
            "drug_protein": dp_stats,
            "protein_disease": pd_stats,
            "protein_protein": ppi_stats,
        },
        **feature_stats,
    }


def normalize_cf_dataset(dataset: str) -> dict:
    ds_dir = DATA_ROOT / dataset
    drug_df = add_or_reset_id_column(pd.read_csv(ds_dir / "drug.csv", dtype=str).fillna(""))
    disease_df = add_or_reset_id_column(pd.read_csv(ds_dir / "disease.csv", dtype=str).fillna(""))
    protein_df = pd.read_csv(ds_dir / "protein.csv", dtype=str).fillna("")
    if protein_df["Sequence"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{dataset} still contains proteins without sequence")
    protein_df = add_or_reset_id_column(protein_df)
    retained_protein_ids = protein_df["Protein"].astype(str).tolist()
    retained_map = {pid: idx for idx, pid in enumerate(retained_protein_ids)}

    validate_dense_matrix(ds_dir / "drug_dis.csv", len(drug_df), len(disease_df))
    old_protein_ids = rebuild_cf_original_protein_ids(dataset)

    dp_src = pd.read_csv(ds_dir / "drug_protein.csv", dtype=str).fillna("")
    pd_src = pd.read_csv(ds_dir / "protein_disease.csv", dtype=str).fillna("")
    ppi_src = pd.read_csv(ds_dir / "protein_protein.csv", dtype=str).fillna("")

    remapped_dp, dp_stats = remap_edge_file_via_old_protein_index(dp_src, ["Protein"], old_protein_ids, retained_map)
    remapped_pd, pd_stats = remap_edge_file_via_old_protein_index(pd_src, ["Protein"], old_protein_ids, retained_map)
    remapped_ppi, ppi_stats = remap_edge_file_via_old_protein_index(
        ppi_src, ["Protein1", "Protein2"], old_protein_ids, retained_map
    )

    drug_df.to_csv(ds_dir / "drug.csv", index=False)
    disease_df.to_csv(ds_dir / "disease.csv", index=False)
    protein_df.to_csv(ds_dir / "protein.csv", index=False)
    remapped_dp.to_csv(ds_dir / "drug_protein.csv", index=False)
    remapped_pd.to_csv(ds_dir / "protein_disease.csv", index=False)
    remapped_ppi.to_csv(ds_dir / "protein_protein.csv", index=False)

    feature_stats = validate_feature_coverage(dataset, drug_df, disease_df, protein_df)
    return {
        "dataset": dataset,
        "n_drug": int(len(drug_df)),
        "n_disease": int(len(disease_df)),
        "n_protein": int(len(protein_df)),
        "recovered_original_protein_index_space": int(len(old_protein_ids)),
        "edge_stats": {
            "drug_protein": dp_stats,
            "protein_disease": pd_stats,
            "protein_protein": ppi_stats,
        },
        **feature_stats,
    }


def validate_dataset(dataset: str) -> dict:
    ds_dir = DATA_ROOT / dataset
    drug_df = pd.read_csv(ds_dir / "drug.csv", dtype=str).fillna("")
    disease_df = pd.read_csv(ds_dir / "disease.csv", dtype=str).fillna("")
    protein_df = pd.read_csv(ds_dir / "protein.csv", dtype=str).fillna("")

    for name, df in [("drug", drug_df), ("disease", disease_df), ("protein", protein_df)]:
        vals = pd.to_numeric(df["ID"], errors="raise").astype(int).tolist()
        if vals != list(range(len(df))):
            raise ValueError(f"{dataset} {name}.csv ID column is not contiguous 0..N-1")

    validate_dense_matrix(ds_dir / "drug_dis.csv", len(drug_df), len(disease_df))

    dp = pd.read_csv(ds_dir / "drug_protein.csv")
    pdg = pd.read_csv(ds_dir / "protein_disease.csv")
    ppi = pd.read_csv(ds_dir / "protein_protein.csv") if (ds_dir / "protein_protein.csv").exists() else None

    validate_index_range(dp["Drug"], len(drug_df), f"{dataset} drug_protein.Drug")
    validate_index_range(dp["Protein"], len(protein_df), f"{dataset} drug_protein.Protein")
    validate_index_range(pdg["Protein"], len(protein_df), f"{dataset} protein_disease.Protein")
    validate_index_range(pdg["Disease"], len(disease_df), f"{dataset} protein_disease.Disease")
    if ppi is not None:
        validate_index_range(ppi["Protein1"], len(protein_df), f"{dataset} protein_protein.Protein1")
        validate_index_range(ppi["Protein2"], len(protein_df), f"{dataset} protein_protein.Protein2")

    validate_feature_coverage(dataset, drug_df, disease_df, protein_df)
    return {
        "dataset": dataset,
        "n_drug": int(len(drug_df)),
        "n_disease": int(len(disease_df)),
        "n_protein": int(len(protein_df)),
        "n_drug_protein": int(len(dp)),
        "n_protein_disease": int(len(pdg)),
        "n_protein_protein": int(len(ppi)) if ppi is not None else 0,
        "mapping_related_drop_count": 0,
        "runtime_missing_embedding_count": 0,
    }


def main() -> None:
    if not REDDA_B_ROOT.exists():
        raise FileNotFoundError(REDDA_B_ROOT)
    for needed in [
        CTD_ROOT / "CTD_chemicals.tsv.gz",
        CTD_ROOT / "CTD_chem_gene_ixns.tsv.gz",
        CTD_ROOT / "CTD_genes.tsv.gz",
        CTD_ROOT / "CTD_genes_diseases.tsv.gz",
    ]:
        if not needed.exists():
            raise FileNotFoundError(needed)

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    normalize_report = [
        normalize_bdataset(),
        normalize_cf_dataset("Cdataset"),
        normalize_cf_dataset("Fdataset"),
    ]
    validation_report = [
        validate_dataset("Bdataset"),
        validate_dataset("Cdataset"),
        validate_dataset("Fdataset"),
    ]

    (REPORT_ROOT / "training_input_normalization_report.json").write_text(
        json.dumps(normalize_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (REPORT_ROOT / "training_input_validation_report.json").write_text(
        json.dumps(validation_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"normalization": normalize_report, "validation": validation_report}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
