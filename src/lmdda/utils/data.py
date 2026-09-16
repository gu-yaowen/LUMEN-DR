from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


@dataclass
class DatasetBundle:
    dataset: str
    drug_df: pd.DataFrame
    disease_df: pd.DataFrame
    protein_df: pd.DataFrame
    dd_pairs: pd.DataFrame
    graph: HeteroData


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"{name}: missing columns {missing}")


def _require_contiguous_ids(df: pd.DataFrame, name: str) -> None:
    _require_columns(df, ["ID"], name)
    vals = pd.to_numeric(df["ID"], errors="raise").astype(np.int64).tolist()
    expected = list(range(len(df)))
    if vals != expected:
        raise ValueError(f"{name}: ID column must be contiguous 0..N-1")


def _expand_drug_disease_matrix(path: Path, n_drug: int, n_disease: int) -> pd.DataFrame:
    mat = pd.read_csv(path, header=None).to_numpy(dtype=np.int64)
    if mat.shape != (n_drug, n_disease):
        raise ValueError(f"{path}: matrix shape {mat.shape} != {(n_drug, n_disease)}")
    drug_idx, disease_idx = np.indices(mat.shape)
    return pd.DataFrame(
        {
            "drug_idx": drug_idx.ravel().astype(np.int64),
            "disease_idx": disease_idx.ravel().astype(np.int64),
            "label": mat.ravel().astype(np.int64),
        }
    )


def _load_index_edge(
    path: Path,
    src_col: str,
    dst_col: str,
    src_limit: int,
    dst_limit: int,
    name: str,
) -> torch.Tensor:
    df = pd.read_csv(path)
    _require_columns(df, [src_col, dst_col], name)
    src = pd.to_numeric(df[src_col], errors="raise").to_numpy(dtype=np.int64)
    dst = pd.to_numeric(df[dst_col], errors="raise").to_numpy(dtype=np.int64)
    if len(src):
        if src.min() < 0 or src.max() >= src_limit:
            raise ValueError(f"{name}: {src_col} out of range [0, {src_limit})")
        if dst.min() < 0 or dst.max() >= dst_limit:
            raise ValueError(f"{name}: {dst_col} out of range [0, {dst_limit})")
    return torch.tensor(np.vstack([src, dst]), dtype=torch.long)


def load_dataset_bundle(data_root: Path, dataset: str, add_ppi_reverse: bool = True) -> DatasetBundle:
    ds_dir = data_root / dataset

    drug_df = pd.read_csv(ds_dir / "drug.csv")
    disease_df = pd.read_csv(ds_dir / "disease.csv")
    protein_df = pd.read_csv(ds_dir / "protein.csv")

    _require_columns(drug_df, ["Drug"], f"{dataset}.drug")
    _require_columns(disease_df, ["Disease"], f"{dataset}.disease")
    _require_columns(protein_df, ["Protein"], f"{dataset}.protein")
    _require_contiguous_ids(drug_df, f"{dataset}.drug")
    _require_contiguous_ids(disease_df, f"{dataset}.disease")
    _require_contiguous_ids(protein_df, f"{dataset}.protein")

    hetero = HeteroData()
    hetero["drug"].num_nodes = len(drug_df)
    hetero["disease"].num_nodes = len(disease_df)
    hetero["protein"].num_nodes = len(protein_df)

    hetero["drug", "to_protein", "protein"].edge_index = _load_index_edge(
        ds_dir / "drug_protein.csv",
        "Drug",
        "Protein",
        len(drug_df),
        len(protein_df),
        f"{dataset}.drug_protein",
    )
    hetero["protein", "to_drug", "drug"].edge_index = _load_index_edge(
        ds_dir / "drug_protein.csv",
        "Protein",
        "Drug",
        len(protein_df),
        len(drug_df),
        f"{dataset}.drug_protein",
    )

    hetero["protein", "to_disease", "disease"].edge_index = _load_index_edge(
        ds_dir / "protein_disease.csv",
        "Protein",
        "Disease",
        len(protein_df),
        len(disease_df),
        f"{dataset}.protein_disease",
    )
    hetero["disease", "to_protein", "protein"].edge_index = _load_index_edge(
        ds_dir / "protein_disease.csv",
        "Disease",
        "Protein",
        len(disease_df),
        len(protein_df),
        f"{dataset}.protein_disease",
    )

    ppi_path = ds_dir / "protein_protein.csv"
    if ppi_path.exists():
        hetero["protein", "to_protein", "protein"].edge_index = _load_index_edge(
            ppi_path,
            "Protein1",
            "Protein2",
            len(protein_df),
            len(protein_df),
            f"{dataset}.protein_protein",
        )
        if add_ppi_reverse:
            hetero["protein", "to_protein_rev", "protein"].edge_index = _load_index_edge(
                ppi_path,
                "Protein2",
                "Protein1",
                len(protein_df),
                len(protein_df),
                f"{dataset}.protein_protein",
            )

    dd_pairs = _expand_drug_disease_matrix(ds_dir / "drug_dis.csv", len(drug_df), len(disease_df))
    return DatasetBundle(
        dataset=dataset,
        drug_df=drug_df,
        disease_df=disease_df,
        protein_df=protein_df,
        dd_pairs=dd_pairs,
        graph=hetero,
    )
