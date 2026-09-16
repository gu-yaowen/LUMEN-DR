from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .losses import symmetric_inbatch_infonce_loss
from .models import LmddaModel
from .utils.checkpoint import load_checkpoint, save_checkpoint
from .utils.data import DatasetBundle, load_dataset_bundle
from .utils.features import load_feature_tensors
from .utils.io import ensure_dir, write_csv, write_json
from .utils.logger import build_logger
from .utils.metrics import classification_metrics
from .utils.plot import plot_training_curve
from .utils.scheduler import build_scheduler, scheduler_step
from .utils.seed import set_seed
from .utils.splits import build_folds


def _to_device_feature_dict(
    feature_tensors: dict[str, torch.Tensor | None],
    device: torch.device,
) -> dict[str, torch.Tensor | None]:
    return {k: (v.to(device) if v is not None else None) for k, v in feature_tensors.items()}


def _to_device_edge_index_dict(bundle: DatasetBundle, device: torch.device) -> dict[tuple[str, str, str], torch.Tensor]:
    return {k: v.to(device) for k, v in bundle.graph.edge_index_dict.items()}


def _compute_pos_weight(train_df: pd.DataFrame, device: torch.device, enabled: bool) -> torch.Tensor | None:
    if not enabled:
        return None
    y = train_df["label"].to_numpy(dtype=np.int64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    return torch.tensor(float(n_neg) / float(n_pos), dtype=torch.float32, device=device)


def _build_train_pairs(dd_pairs: pd.DataFrame, train_idx: np.ndarray) -> pd.DataFrame:
    return dd_pairs.iloc[train_idx].copy().reset_index(drop=True)


def _split_stats(df: pd.DataFrame) -> dict[str, int]:
    labels = df["label"].to_numpy(dtype=np.int64)
    return {
        "num_pairs": int(len(df)),
        "num_positive": int((labels == 1).sum()),
        "num_negative": int((labels == 0).sum()),
    }


@torch.no_grad()
def _score_pairs(
    model: LmddaModel,
    feature_tensors: dict[str, torch.Tensor | None],
    edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    pair_df: pd.DataFrame,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    x_dict = model.encode_nodes(feature_tensors, edge_index_dict)
    if len(pair_df) == 0:
        return np.array([], dtype=np.float32)
    d_idx = torch.tensor(pair_df["drug_idx"].to_numpy(), dtype=torch.long, device=device)
    s_idx = torch.tensor(pair_df["disease_idx"].to_numpy(), dtype=torch.long, device=device)
    logits_all = model.score_pairs(x_dict, d_idx, s_idx).detach().cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits_all))


def _train_one_epoch(
    model: LmddaModel,
    optimizer: AdamW,
    feature_tensors: dict[str, torch.Tensor | None],
    edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    train_df: pd.DataFrame,
    device: torch.device,
    args,
) -> dict[str, float]:
    model.train()
    pos_weight = _compute_pos_weight(train_df, device, bool(args.use_pos_weight))
    if len(train_df) == 0:
        return {
            "train_loss": math.nan,
            "train_total_loss": math.nan,
            "train_bce_loss": math.nan,
            "train_contrastive_loss": math.nan,
            "train_pos_weight": math.nan,
        }
    d_idx = torch.tensor(train_df["drug_idx"].to_numpy(), dtype=torch.long, device=device)
    s_idx = torch.tensor(train_df["disease_idx"].to_numpy(), dtype=torch.long, device=device)
    y = torch.tensor(train_df["label"].to_numpy(), dtype=torch.float32, device=device)
    x_dict = model.encode_nodes(feature_tensors, edge_index_dict)
    logits = model.score_pairs(x_dict, d_idx, s_idx)
    bce_loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)

    contrastive_loss = logits.new_zeros(())
    if args.use_contrastive:
        pos_mask = y > 0.5
        if int(pos_mask.sum().item()) >= 2:
            pos_drug_idx = d_idx[pos_mask]
            pos_disease_idx = s_idx[pos_mask]
            pos_drug_repr, pos_disease_repr = model.contrastive_embeddings(x_dict, pos_drug_idx, pos_disease_idx)
            contrastive_loss = symmetric_inbatch_infonce_loss(
                pos_drug_repr,
                pos_disease_repr,
                temperature=args.contrastive_temperature,
            )

    loss = bce_loss + args.contrastive_weight * contrastive_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {
        "train_loss": float(loss.item()),
        "train_total_loss": float(loss.item()),
        "train_bce_loss": float(bce_loss.item()),
        "train_contrastive_loss": float(contrastive_loss.item()),
        "train_pos_weight": float(pos_weight.item()) if pos_weight is not None else 1.0,
    }


def _evaluate_fold(
    model: LmddaModel,
    feature_tensors: dict[str, torch.Tensor | None],
    edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    dd_pairs: pd.DataFrame,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float], pd.DataFrame]:
    prob_all = _score_pairs(model, feature_tensors, edge_index_dict, dd_pairs, device)
    pred_df = dd_pairs.copy()
    pred_df["prob"] = prob_all
    pred_df["score"] = np.log(pred_df["prob"] / np.clip(1 - pred_df["prob"], 1e-8, 1.0))
    pred_df["split"] = "train_or_unknown"
    pred_df.loc[val_idx, "split"] = "val"
    pred_df.loc[test_idx, "split"] = "test"

    val = pred_df.iloc[val_idx]
    test = pred_df.iloc[test_idx]
    val_metrics = classification_metrics(val["label"].to_numpy(), val["prob"].to_numpy())
    test_metrics = classification_metrics(test["label"].to_numpy(), test["prob"].to_numpy())
    return val_metrics, test_metrics, pred_df


def run_train_cv(args) -> None:
    set_seed(args.seed)
    run_dir = args.output_root / args.dataset / args.run_name
    ensure_dir(run_dir)
    bundle = load_dataset_bundle(args.data_root, args.dataset)
    feature_sources = {
        "drug": args.drug_feature_source,
        "disease": args.disease_feature_source,
        "protein": args.protein_feature_source,
    }
    base_feature_tensors = load_feature_tensors(args.dataset, args.data_root, args.features_root, feature_sources)
    folds = build_folds(bundle.dd_pairs, args.num_folds, args.seed, args.split_mode)
    if args.fold_index is not None:
        if args.fold_index < 0 or args.fold_index >= len(folds):
            raise ValueError(f"Invalid fold_index={args.fold_index}; expected 0..{len(folds)-1}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    base_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if args.fold_index is None:
        logger = build_logger(run_dir / "train.log")
        write_json(base_config, run_dir / "config.json")
        target_folds = list(enumerate(folds))
    else:
        fold_dir = run_dir / "folds" / f"fold_{args.fold_index}"
        ensure_dir(fold_dir)
        logger = build_logger(fold_dir / "train.log")
        write_json(base_config, fold_dir / "config.json")
        target_folds = [(args.fold_index, folds[args.fold_index])]

    fold_summaries = []
    for fold_id, fold in target_folds:
        fold_dir = run_dir / "folds" / f"fold_{fold_id}"
        ensure_dir(fold_dir)
        logger.info("Fold %d start", fold_id)
        train_stats = _split_stats(bundle.dd_pairs.iloc[fold.train_idx])
        val_stats = _split_stats(bundle.dd_pairs.iloc[fold.val_idx])
        test_stats = _split_stats(bundle.dd_pairs.iloc[fold.test_idx])
        logger.info(
            "fold=%d split_mode=%s train_pairs=%d train_pos=%d val_pairs=%d val_pos=%d test_pairs=%d test_pos=%d",
            fold_id,
            args.split_mode,
            train_stats["num_pairs"],
            train_stats["num_positive"],
            val_stats["num_pairs"],
            val_stats["num_positive"],
            test_stats["num_pairs"],
            test_stats["num_positive"],
        )

        feature_tensors = _to_device_feature_dict(base_feature_tensors, device)
        edge_index_dict = _to_device_edge_index_dict(bundle, device)
        model = LmddaModel(
            feature_dims={k: (v.shape[1] if v is not None else None) for k, v in base_feature_tensors.items()},
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
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = build_scheduler(optimizer, args)

        best_val_aupr = -1.0
        best_epoch = -1
        no_improve = 0
        epoch_rows = []

        for epoch in range(1, args.epochs + 1):
            train_df = _build_train_pairs(bundle.dd_pairs, fold.train_idx)
            train_loss_parts = _train_one_epoch(
                model=model,
                optimizer=optimizer,
                feature_tensors=feature_tensors,
                edge_index_dict=edge_index_dict,
                train_df=train_df,
                device=device,
                args=args,
            )
            val_metrics, test_metrics, _ = _evaluate_fold(
                model=model,
                feature_tensors=feature_tensors,
                edge_index_dict=edge_index_dict,
                dd_pairs=bundle.dd_pairs,
                val_idx=fold.val_idx,
                test_idx=fold.test_idx,
                device=device,
            )
            scheduler_step(scheduler, args, val_metrics["aupr"])
            current_lr = float(optimizer.param_groups[0]["lr"])
            row = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_loss_parts["train_loss"],
                "train_total_loss": train_loss_parts["train_total_loss"],
                "train_bce_loss": train_loss_parts["train_bce_loss"],
                "train_contrastive_loss": train_loss_parts["train_contrastive_loss"],
                "train_pos_weight": train_loss_parts["train_pos_weight"],
                "val_aupr": val_metrics["aupr"],
                "val_auroc": val_metrics["auroc"],
                "test_aupr": test_metrics["aupr"],
                "test_auroc": test_metrics["auroc"],
            }
            epoch_rows.append(row)
            logger.info(
                "fold=%d epoch=%d lr=%.8f total=%.6f bce=%.6f contrast=%.6f pos_weight=%.3f val_aupr=%.6f val_auroc=%.6f",
                fold_id,
                epoch,
                current_lr,
                train_loss_parts["train_total_loss"],
                train_loss_parts["train_bce_loss"],
                train_loss_parts["train_contrastive_loss"],
                train_loss_parts["train_pos_weight"],
                val_metrics["aupr"],
                val_metrics["auroc"],
            )

            if val_metrics["aupr"] > best_val_aupr:
                best_val_aupr = val_metrics["aupr"]
                best_epoch = epoch
                no_improve = 0
                save_checkpoint(
                    fold_dir / "best.ckpt",
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": vars(args),
                "fold_id": fold_id,
                "split_mode": args.split_mode,
                "best_epoch": best_epoch,
            },
        )
            else:
                no_improve += 1
            if no_improve >= args.patience:
                logger.info("fold=%d early stop at epoch=%d", fold_id, epoch)
                break

        save_checkpoint(
            fold_dir / "last.ckpt",
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "fold_id": fold_id,
                "split_mode": args.split_mode,
            },
        )

        best_ckpt = load_checkpoint(fold_dir / "best.ckpt", map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        val_metrics, test_metrics, pred_df = _evaluate_fold(
            model=model,
            feature_tensors=feature_tensors,
            edge_index_dict=edge_index_dict,
            dd_pairs=bundle.dd_pairs,
            val_idx=fold.val_idx,
            test_idx=fold.test_idx,
            device=device,
        )

        epoch_df = pd.DataFrame(epoch_rows)
        write_csv(epoch_df, fold_dir / "epoch_metrics.csv")
        plot_training_curve(epoch_df, fold_dir / "train_curve.png")

        write_csv(pred_df[pred_df["split"] == "val"], fold_dir / "val_predictions.csv")
        write_csv(pred_df[pred_df["split"] == "test"], fold_dir / "test_predictions.csv")
        write_csv(pred_df, fold_dir / "test_pair_scores_full.csv")

        fold_metrics = {f"val_{k}": v for k, v in val_metrics.items()} | {f"test_{k}": v for k, v in test_metrics.items()}
        fold_metrics["best_epoch"] = best_epoch
        write_json(fold_metrics, fold_dir / "metrics.json")
        fold_summaries.append({"fold": fold_id, **fold_metrics})

    cv_df = pd.DataFrame(fold_summaries)
    if args.fold_index is None:
        write_csv(cv_df, run_dir / "cv_summary.csv")
        summary = {
            "dataset": args.dataset,
            "run_name": args.run_name,
            "num_folds": args.num_folds,
            "mean_metrics": {k: float(cv_df[k].mean()) for k in cv_df.columns if k != "fold"},
        }
        write_json(summary, run_dir / "cv_summary.json")
    else:
        write_csv(cv_df, fold_dir / "fold_cv_summary.csv")
