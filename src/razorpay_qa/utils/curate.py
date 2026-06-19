"""Dataset curation: balance checks, review sampling, splits, freeze, and versioned output."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .. import __version__
from .config import DATASET_DIR, REPO_ROOT, Settings
from .schema import Category, QARecord, ReviewStatus


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_provenance() -> dict[str, str | None]:
    """Best-effort git commit/branch; ``None`` when not a repo or git is absent."""
    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() or None if out.returncode == 0 else None

    return {
        "git_commit": _run("rev-parse", "--short", "HEAD"),
        "git_branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
    }


def update_version_registry(entry: dict, dataset_dir: Path) -> Path:
    """Append one entry to ``artifacts/dataset/versions.json`` (read-modify-write)."""
    path = dataset_dir / "versions.json"
    registry: list = []
    if path.exists():
        try:
            registry = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(registry, list):
                registry = []
        except (ValueError, OSError):
            registry = []
    registry.append(entry)
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return path


def write_latest_pointer(tag: str, dataset_dir: Path) -> Path:
    path = dataset_dir / "LATEST"
    path.write_text(tag + "\n", encoding="utf-8")
    return path


def _next_version(dataset_dir: Path) -> int:
    """Next dataset version by scanning ``artifacts/dataset/v<N>/`` subfolders."""
    versions = []
    if dataset_dir.exists():
        for p in dataset_dir.glob("v*"):
            if p.is_dir():
                try:
                    versions.append(int(p.name[1:]))
                except ValueError:
                    continue
    return (max(versions) + 1) if versions else 1


def mark_review_sample(records: list[QARecord], settings: Settings) -> None:
    """Flag a deterministic per-category sample for human spot-check."""
    n = int(settings.pipeline.get("review", {}).get("spot_check_per_category", 3))
    rng = random.Random(settings.seed)
    by_cat: dict[Category, list[QARecord]] = {Category.A: [], Category.B: [], Category.C: []}
    for r in records:
        by_cat[r.category].append(r)
    for _cat, recs in by_cat.items():
        candidates = [r for r in recs if r.review_status != ReviewStatus.human_verified]
        rng.shuffle(candidates)
        for r in candidates[:n]:
            r.review_status = ReviewStatus.spot_checked


def assign_splits(records: list[QARecord], settings: Settings) -> dict[str, list[str]]:
    """Stratified train/val/test split by (category, question_type). Deterministic."""
    cfg = settings.pipeline.get("split", {})
    train_p = float(cfg.get("train", 0.7))
    val_p = float(cfg.get("val", 0.15))
    rng = random.Random(settings.seed + 1)

    strata: dict[tuple, list[QARecord]] = {}
    for r in records:
        strata.setdefault((r.category.value, r.question_type.value), []).append(r)

    splits = {"train": [], "val": [], "test": []}
    for _, recs in sorted(strata.items()):
        recs = sorted(recs, key=lambda r: r.id)
        rng.shuffle(recs)
        n = len(recs)
        n_train = round(n * train_p)
        n_val = round(n * val_p)
        for i, r in enumerate(recs):
            if i < n_train:
                splits["train"].append(r.id)
            elif i < n_train + n_val:
                splits["val"].append(r.id)
            else:
                splits["test"].append(r.id)
    return splits


def freeze_dataset(records: list[QARecord], version: int, version_dir: Path) -> Path:
    version_dir.mkdir(parents=True, exist_ok=True)
    tag = f"v{version}"
    path = version_dir / "dataset.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            r.dataset_version = tag
            fh.write(r.model_dump_json(exclude_none=False) + "\n")
    latest = DATASET_DIR / "dataset.jsonl"
    latest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def write_dataset_card(records: list[QARecord], version: int, splits: dict, out_dir: Path) -> Path:
    cat = Counter(r.category.value for r in records)
    topic = Counter(r.topic for r in records)
    qtype = Counter(r.question_type.value for r in records)
    diff = Counter(r.difficulty.value for r in records)
    reviewed = sum(1 for r in records if r.review_status != ReviewStatus.unreviewed)

    def fmt(counter: Counter) -> str:
        return ", ".join(f"`{k}`: {v}" for k, v in sorted(counter.items()))

    lines = [
        f"# Dataset Card — Razorpay ToS Q&A (v{version})",
        "",
        f"- **Records:** {len(records)}",
        f"- **Source:** `{records[0].source_doc}` (sha256 `{records[0].source_hash[:12]}…`)",
        f"- **Reviewed fraction:** {reviewed}/{len(records)} "
        f"({reviewed / len(records):.0%}) carry a review_status above `unreviewed`",
        "",
        "## Category distribution",
        fmt(cat),
        "",
        "## Topic coverage",
        fmt(topic),
        "",
        "## Question-type distribution",
        fmt(qtype),
        "",
        "## Difficulty distribution",
        fmt(diff),
        "",
        "## Splits (stratified by category × question_type)",
        f"train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}",
        "",
        "## Known limitations",
        "- Clause numbering in this PDF differs from Razorpay's live site; citations are "
        "grounded to the PDF's actual clauses (verified by substring check).",
        "- Questions/answers are LLM-generated (DeepEval Synthesizer + enrichment); re-running "
        "with a different provider/model/seed yields different phrasings (see README determinism).",
        "- Short verbatim quotes are used for grounding only (Razorpay's ToS is copyrighted).",
        "",
    ]
    path = out_dir / "dataset_card.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_run_manifest(records: list[QARecord], version: int, settings: Settings,
                       provider: str, splits: dict, postprocess_report, out_dir: Path,
                       dataset_sha256: str, git: dict) -> Path:
    cat = Counter(r.category.value for r in records)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool_version": __version__,
        "dataset_version": f"v{version}",
        "schema_version": settings.schema_version,
        "dataset_sha256": dataset_sha256,
        "git_commit": git.get("git_commit"),
        "git_branch": git.get("git_branch"),
        "provider": provider,
        "model": records[0].provenance.model if records else None,
        "seed": settings.seed,
        "source_doc": records[0].source_doc if records else None,
        "source_hash": records[0].source_hash if records else None,
        "totals": {"records": len(records), "by_category": dict(cat)},
        "min_per_category": settings.min_per_category,
        "postprocess": {
            "kept": postprocess_report.kept,
            "dropped_citation": postprocess_report.dropped_citation,
            "dropped_duplicate": postprocess_report.dropped_duplicate,
            "dropped_language": postprocess_report.dropped_language,
            "dropped_guardrail": postprocess_report.dropped_guardrail,
            "guardrail_flagged": postprocess_report.guardrail_flagged,
        },
        "splits": {k: len(v) for k, v in splits.items()},
        "config": {
            "pipeline": settings.pipeline,
        },
    }
    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def check_balance(records: list[QARecord], settings: Settings) -> dict[str, int]:
    cat = Counter(r.category.value for r in records)
    shortfall = {
        c: settings.min_per_category - cat.get(c, 0)
        for c in ("A", "B", "C")
        if cat.get(c, 0) < settings.min_per_category
    }
    if shortfall:
        raise ValueError(
            f"Category balance check failed; need >= {settings.min_per_category} each. "
            f"Counts={dict(cat)}, shortfall={shortfall}"
        )
    return dict(cat)


def curate(records: list[QARecord], settings: Settings, provider: str, postprocess_report) -> dict:
    check_balance(records, settings)
    mark_review_sample(records, settings)
    splits = assign_splits(records, settings)
    version = _next_version(DATASET_DIR)
    version_dir = DATASET_DIR / f"v{version}"
    version_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = freeze_dataset(records, version, version_dir)
    dataset_sha256 = _sha256_file(dataset_path)
    git = _git_provenance()
    card_path = write_dataset_card(records, version, splits, version_dir)
    manifest_path = write_run_manifest(
        records, version, settings, provider, splits, postprocess_report, version_dir,
        dataset_sha256, git,
    )

    tag = f"v{version}"
    cat = Counter(r.category.value for r in records)
    update_version_registry(
        {
            "version": tag,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset_sha256": dataset_sha256,
            "total": len(records),
            "by_category": dict(cat),
            "source_hash": records[0].source_hash if records else None,
            "tool_version": __version__,
            "schema_version": settings.schema_version,
            **git,
        },
        DATASET_DIR,
    )
    write_latest_pointer(tag, DATASET_DIR)

    return {
        "version": version,
        "dataset_path": str(dataset_path),
        "card_path": str(card_path),
        "manifest_path": str(manifest_path),
        "dataset_sha256": dataset_sha256,
        "splits": splits,
    }
