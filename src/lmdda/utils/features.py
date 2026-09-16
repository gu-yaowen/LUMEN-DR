from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _load_pkl_dict(path: Path) -> dict[str, list[float]]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, got {type(obj)}")
    return obj


def _align_entity_embeddings(entity_ids: list[str], emb_dict: dict[str, list[float]], name: str) -> torch.Tensor:
    if len(emb_dict) == 0:
        raise ValueError(f"{name}: empty embedding dict")
    emb_dim = len(next(iter(emb_dict.values())))
    missing = [eid for eid in entity_ids if eid not in emb_dict]
    if missing:
        raise ValueError(f"{name}: {len(missing)} missing embeddings, first={missing[:5]}")
    arr = np.asarray([emb_dict[eid] for eid in entity_ids], dtype=np.float32)
    return torch.from_numpy(arr)


def load_feature_tensors(
    dataset: str,
    data_root: Path,
    features_root: Path,
    feature_sources: dict[str, str] | None = None,
) -> dict[str, torch.Tensor | None]:
    ds_dir = data_root / dataset
    feat_dir = features_root / dataset
    feature_sources = feature_sources or {
        "drug": "pretrained",
        "disease": "pretrained",
        "protein": "pretrained",
    }

    drug_df = pd.read_csv(ds_dir / "drug.csv")
    disease_df = pd.read_csv(ds_dir / "disease.csv")
    protein_df = pd.read_csv(ds_dir / "protein.csv")

    drug_ids = drug_df["Drug"].astype(str).tolist()
    disease_ids = disease_df["Disease"].astype(str).tolist()
    protein_ids = protein_df["Protein"].astype(str).tolist()

    out: dict[str, torch.Tensor | None] = {}
    if feature_sources["drug"] == "pretrained":
        drug_emb = _load_pkl_dict(feat_dir / "MolFormer_drug_emb.pkl")
        out["drug"] = _align_entity_embeddings(drug_ids, drug_emb, "drug")
    else:
        out["drug"] = None
    if feature_sources["disease"] == "pretrained":
        disease_emb = _load_pkl_dict(feat_dir / "BERT_disease_emb.pkl")
        out["disease"] = _align_entity_embeddings(disease_ids, disease_emb, "disease")
    else:
        out["disease"] = None
    if feature_sources["protein"] == "pretrained":
        protein_emb = _load_pkl_dict(feat_dir / "ESMC_protein_emb.pkl")
        out["protein"] = _align_entity_embeddings(protein_ids, protein_emb, "protein")
    else:
        out["protein"] = None
    return out
