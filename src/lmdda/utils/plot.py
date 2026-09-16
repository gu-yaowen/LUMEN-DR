from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curve(epoch_metrics: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epoch_metrics["epoch"], epoch_metrics["train_loss"], label="train_loss")
    if "val_aupr" in epoch_metrics.columns:
        ax2 = ax.twinx()
        ax2.plot(epoch_metrics["epoch"], epoch_metrics["val_aupr"], color="tab:orange", label="val_aupr")
        ax2.set_ylabel("val_aupr")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train_loss")
    ax.set_title("Training Curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

