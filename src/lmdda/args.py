from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("LUMEN-DR")
    parser.add_argument("--mode", choices=["train", "predict", "export_embeddings"], default="train")
    parser.add_argument("--dataset", choices=["Bdataset", "Cdataset", "Fdataset"], default="Bdataset")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--features-root", default="features")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--run-name", default="hgt_main")

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int, default=None)
    parser.add_argument("--split-mode", choices=["pair", "cold_drug", "cold_disease"], default="pair")

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--proj-norm-type", choices=["layernorm", "batchnorm", "none"], default="layernorm")
    parser.add_argument("--activation", choices=["relu", "gelu", "leaky_relu"], default="relu")
    parser.add_argument("--drug-feature-source", choices=["pretrained", "id"], default="pretrained")
    parser.add_argument("--disease-feature-source", choices=["pretrained", "id"], default="pretrained")
    parser.add_argument("--protein-feature-source", choices=["pretrained", "id"], default="pretrained")
    parser.add_argument("--use-pos-weight", type=int, choices=[0, 1], default=1)
    parser.add_argument("--use-contrastive", type=int, choices=[0, 1], default=0)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--contrastive-temperature", type=float, default=0.2)
    parser.add_argument("--contrastive-dim", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--scheduler", choices=["none", "cosine", "linear", "step", "plateau"], default="none")
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--scheduler-t-max", type=int, default=0)
    parser.add_argument("--scheduler-step-size", type=int, default=0)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--neg-ratio", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=8)

    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--candidate-csv", default=None)
    parser.add_argument("--predict-split", choices=["all", "val", "test"], default="all")
    return parser


def parse_args() -> argparse.Namespace:
    args = build_parser().parse_args()
    args.data_root = Path(args.data_root)
    args.features_root = Path(args.features_root)
    args.output_root = Path(args.output_root)
    if args.checkpoint is not None:
        args.checkpoint = Path(args.checkpoint)
    if args.candidate_csv is not None:
        args.candidate_csv = Path(args.candidate_csv)
    return args
