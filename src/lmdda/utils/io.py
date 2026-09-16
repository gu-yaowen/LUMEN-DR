from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)

