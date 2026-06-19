from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"

ARTIFACTS = REPO_ROOT / "artifacts"
SOURCE_DIR = ARTIFACTS / "source"
DATASET_DIR = ARTIFACTS / "dataset"
EVAL_DIR = ARTIFACTS / "eval"


@dataclass
class Settings:
    pipeline: dict[str, Any] = field(default_factory=dict)
    taxonomy: dict[str, Any] = field(default_factory=dict)

    @property
    def pdf_path(self) -> Path:
        return REPO_ROOT / self.pipeline["source"]["pdf_path"]

    @property
    def jurisdiction(self) -> str:
        return str(self.pipeline["source"].get("jurisdiction", "India"))

    @property
    def tos_effective_date(self):
        return self.pipeline["source"].get("tos_effective_date")

    @property
    def seed(self) -> int:
        return int(self.pipeline["generation"]["seed"])

    @property
    def schema_version(self) -> str:
        return str(self.pipeline["generation"]["schema_version"])

    @property
    def min_per_category(self) -> int:
        return int(self.pipeline["generation"]["min_per_category"])

    @property
    def pass_threshold(self) -> float:
        return float(self.pipeline["evaluation"]["pass_threshold"])


def load_settings() -> Settings:
    pipeline = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text(encoding="utf-8"))
    taxonomy = yaml.safe_load((CONFIG_DIR / "taxonomy.yaml").read_text(encoding="utf-8"))
    return Settings(pipeline=pipeline, taxonomy=taxonomy)
