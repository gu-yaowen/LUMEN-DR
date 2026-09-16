from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .io import ensure_dir


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    torch.save(state, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def remap_legacy_model_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    remapped = dict(state_dict)
    legacy_prefix_map = {
        "drug_proj.": "drug_encoder.pretrained_proj.",
        "disease_proj.": "disease_encoder.pretrained_proj.",
        "protein_proj.": "protein_encoder.pretrained_proj.",
    }
    for legacy_prefix, new_prefix in legacy_prefix_map.items():
        legacy_keys = [k for k in remapped.keys() if k.startswith(legacy_prefix)]
        for legacy_key in legacy_keys:
            new_key = new_prefix + legacy_key[len(legacy_prefix) :]
            remapped.setdefault(new_key, remapped[legacy_key])
            remapped.pop(legacy_key, None)
    return remapped
