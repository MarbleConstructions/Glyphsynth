from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Settings:
    seed: int = 42
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    samples_per_digit: int = 300
    validation_share: float = 0.2
    patience: int = 5
    image_side: int = 28
    num_classes: int = 10
    weights_path: str = "artifacts/glyphsort.pt"


def load_settings(path: str | Path = "config.yaml") -> Settings:
    base = Settings()
    file = Path(path)
    if not file.exists():
        return base

    raw: dict[str, Any] = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    known = {f.name for f in fields(Settings)}
    filtered = {k: v for k, v in raw.items() if k in known}
    merged = {**base.__dict__, **filtered}
    return Settings(**merged)from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Settings:
    seed: int = 42
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    samples_per_digit: int = 300
    validation_share: float = 0.2
    patience: int = 5
    image_side: int = 28
    num_classes: int = 10
    weights_path: str = "artifacts/glyphsort.pt"


def load_settings(path: str | Path = "config.yaml") -> Settings:
    base = Settings()
    file = Path(path)
    if not file.exists():
        return base

    raw: dict[str, Any] = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    known = {f.name for f in fields(Settings)}
    filtered = {k: v for k, v in raw.items() if k in known}
    merged = {**base.__dict__, **filtered}
    return Settings(**merged)