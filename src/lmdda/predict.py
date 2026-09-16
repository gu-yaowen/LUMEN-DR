from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .models import LmddaModel
from .utils.checkpoint import load_checkpoint
from .utils.data import load_dataset_bundle
from .utils.features import load_feature_tensors
from .utils.io import ensure_dir, write_csv


@torch.no_grad()
def _score_pairs(model, feature_tensors, edge_index_dict, pair_df, batch_size, device):
    model.eval()
    outs = []
    x_dict = model.encode_nodes(feature_tensors, edge_index_dict)
    for start in range(0, len(pair_df), batch_size):
        b = pair_df.iloc[start : start + batch_size]
        d_idx = torch.tensor(b["drug_idx"].to_numpy(), dtype=torch.long, device=device)
        s_idx = torch.tensor(b["disease_idx"].to_numpy(), dtype=torch.long, device=device)
        logits = model.score_pairs(x_dict, d_idx, s_idx)
        outs.append(logits.detach().cpu().numpy())
    logits = np.concatenate(outs, axis=0) if outs else np.array([], dtype=np.float32)
    prob = 1.0 / (1.0 + np.exp(-logits))
    return logits, prob


def run_predict(args) -> None:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required in predict mode")
    bundle = load_dataset_bundle(args.data_root, args.dataset)
    feature_sources = {
        "drug": args.drug_feature_source,
        "disease": args.disease_feature_source,
        "protein": args.protein_feature_source,
    }
    feature_tensors = load_feature_tensors(args.dataset, args.data_root, args.features_root, feature_sources)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = LmddaModel(
        feature_dims={k: (v.shape[1] if v is not None else None) for k, v in feature_tensors.items()},
        node_counts={
            "drug": bundle.graph["drug"].num_nodes,
            "disease": bundle.graph["disease"].num_nodes,
            "protein": bundle.graph["protein"].num_nodes,
        },
        feature_sources=feature_sources,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        contrastive_dim=args.contrastive_dim,
        proj_norm_type=args.proj_norm_type,
        activation=args.activation,
        metadata=bundle.graph.metadata(),
    ).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    feature_tensors = {k: (v.to(device) if v is not None else None) for k, v in feature_tensors.items()}
    edge_index_dict = {k: v.to(device) for k, v in bundle.graph.edge_index_dict.items()}

    if args.candidate_csv is not None:
        pair_df = pd.read_csv(args.candidate_csv)
    else:
        pair_df = bundle.dd_pairs.copy()

    logits, prob = _score_pairs(model, feature_tensors, edge_index_dict, pair_df, args.batch_size, device)
    out = pair_df.copy()
    out["score"] = logits
    out["prob"] = prob

    out_dir = args.output_root / args.dataset / args.run_name / "predict"
    ensure_dir(out_dir)
    write_csv(out, out_dir / "predictions.csv")
