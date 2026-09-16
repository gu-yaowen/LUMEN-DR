from __future__ import annotations

from pathlib import Path

import torch

from .models import LmddaModel
from .utils.checkpoint import load_checkpoint, remap_legacy_model_state_dict
from .utils.data import load_dataset_bundle
from .utils.features import load_feature_tensors
from .utils.io import ensure_dir


@torch.no_grad()
def run_export_embeddings(args) -> None:
    bundle = load_dataset_bundle(args.data_root, args.dataset)
    feature_sources = {
        "drug": args.drug_feature_source,
        "disease": args.disease_feature_source,
        "protein": args.protein_feature_source,
    }
    feature_tensors = load_feature_tensors(args.dataset, args.data_root, args.features_root, feature_sources)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_dir = args.output_root / args.dataset / args.run_name

    target_folds = range(args.num_folds) if args.fold_index is None else [args.fold_index]
    for fold_id in target_folds:
        ckpt_path = run_dir / "folds" / f"fold_{fold_id}" / "best.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

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
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        model_state_dict = remap_legacy_model_state_dict(ckpt["model_state_dict"])
        model.load_state_dict(model_state_dict)
        model.eval()

        feature_tensors_device = {k: (v.to(device) if v is not None else None) for k, v in feature_tensors.items()}
        edge_index_dict = {k: v.to(device) for k, v in bundle.graph.edge_index_dict.items()}
        x_dict = model.encode_nodes(feature_tensors_device, edge_index_dict)

        export = {
            "dataset": args.dataset,
            "run_name": args.run_name,
            "fold_id": fold_id,
            "drug_backbone": x_dict["drug"].detach().cpu(),
            "disease_backbone": x_dict["disease"].detach().cpu(),
            "protein_backbone": x_dict["protein"].detach().cpu(),
            "drug_contrastive": model.drug_contrast_proj(x_dict["drug"]).detach().cpu(),
            "disease_contrastive": model.disease_contrast_proj(x_dict["disease"]).detach().cpu(),
            "drug_entity_ids": bundle.drug_df["Drug"].tolist(),
            "disease_entity_ids": bundle.disease_df["Disease"].tolist(),
            "protein_entity_ids": bundle.protein_df["Protein"].tolist(),
        }

        out_path = run_dir / "exports" / f"fold_{fold_id}_embeddings.pt"
        ensure_dir(out_path.parent)
        torch.save(export, out_path)
