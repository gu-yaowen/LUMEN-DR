from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


@dataclass
class FoldSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def build_stratified_folds(dd_pairs: pd.DataFrame, num_folds: int, seed: int) -> list[FoldSplit]:
    y = dd_pairs["label"].to_numpy()
    idx = np.arange(len(dd_pairs))
    if num_folds == 1:
        train_idx, heldout_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
        y_heldout = y[heldout_idx]
        val_idx, test_idx = train_test_split(heldout_idx, test_size=0.5, stratify=y_heldout, random_state=seed + 777)
        return [FoldSplit(train_idx=np.asarray(train_idx), val_idx=np.asarray(val_idx), test_idx=np.asarray(test_idx))]
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds = []
    for train_val_idx, test_idx in skf.split(idx, y):
        y_train_val = y[train_val_idx]
        skf_inner = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed + 777)
        inner_train_rel, inner_val_rel = next(skf_inner.split(train_val_idx, y_train_val))
        train_idx = train_val_idx[inner_train_rel]
        val_idx = train_val_idx[inner_val_rel]
        folds.append(FoldSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx))
    return folds


def _build_entity_cold_folds(
    dd_pairs: pd.DataFrame,
    entity_col: str,
    num_folds: int,
    seed: int,
) -> list[FoldSplit]:
    entity_ids = np.sort(dd_pairs[entity_col].unique())
    if num_folds == 1:
        train_entities, heldout_entities = train_test_split(entity_ids, test_size=0.2, random_state=seed)
        val_entities, test_entities = train_test_split(heldout_entities, test_size=0.5, random_state=seed + 777)
        train_mask = dd_pairs[entity_col].isin(train_entities).to_numpy()
        val_mask = dd_pairs[entity_col].isin(val_entities).to_numpy()
        test_mask = dd_pairs[entity_col].isin(test_entities).to_numpy()
        return [
            FoldSplit(
                train_idx=np.flatnonzero(train_mask),
                val_idx=np.flatnonzero(val_mask),
                test_idx=np.flatnonzero(test_mask),
            )
        ]

    outer_kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds = []
    for train_val_rel, test_rel in outer_kf.split(entity_ids):
        train_val_entities = entity_ids[train_val_rel]
        test_entities = entity_ids[test_rel]
        inner_kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed + 777)
        inner_train_rel, inner_val_rel = next(inner_kf.split(train_val_entities))
        train_entities = train_val_entities[inner_train_rel]
        val_entities = train_val_entities[inner_val_rel]

        train_mask = dd_pairs[entity_col].isin(train_entities).to_numpy()
        val_mask = dd_pairs[entity_col].isin(val_entities).to_numpy()
        test_mask = dd_pairs[entity_col].isin(test_entities).to_numpy()
        folds.append(
            FoldSplit(
                train_idx=np.flatnonzero(train_mask),
                val_idx=np.flatnonzero(val_mask),
                test_idx=np.flatnonzero(test_mask),
            )
        )
    return folds


def build_folds(dd_pairs: pd.DataFrame, num_folds: int, seed: int, split_mode: str) -> list[FoldSplit]:
    if split_mode == "pair":
        return build_stratified_folds(dd_pairs, num_folds, seed)
    if split_mode == "cold_drug":
        return _build_entity_cold_folds(dd_pairs, "drug_idx", num_folds, seed)
    if split_mode == "cold_disease":
        return _build_entity_cold_folds(dd_pairs, "disease_idx", num_folds, seed)
    raise ValueError(f"Unsupported split_mode: {split_mode}")
