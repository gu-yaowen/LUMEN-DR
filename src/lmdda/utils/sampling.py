from __future__ import annotations

import numpy as np
import pandas as pd


def sample_train_pairs(dd_pairs: pd.DataFrame, train_idx: np.ndarray, neg_ratio: float, seed: int) -> pd.DataFrame:
    train_df = dd_pairs.iloc[train_idx].copy()
    pos_df = train_df[train_df["label"] == 1]
    neg_df = train_df[train_df["label"] == 0]
    n_pos = len(pos_df)
    n_neg_keep = min(len(neg_df), int(round(neg_ratio * n_pos)))
    if n_neg_keep > 0:
        neg_keep = neg_df.sample(n=n_neg_keep, random_state=seed)
        out = pd.concat([pos_df, neg_keep], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        out = pos_df.reset_index(drop=True)
    return out

