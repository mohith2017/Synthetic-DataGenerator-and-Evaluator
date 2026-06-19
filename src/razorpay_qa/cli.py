from __future__ import annotations

import argparse
import sys

from .generation import postprocess
from .utils.config import DATASET_DIR, REPO_ROOT, SOURCE_DIR, load_settings
from .utils.curate import curate
from .utils.ingest import build_clause_index, load_source, save_parsed_text
from .utils.llm import PROVIDER


def _load_env() -> None:
    """Load ``.env`` from the repo root so ANTHROPIC_API_KEY is picked up without exporting."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env")


def _cmd_generate(args) -> int:
    settings = load_settings()
    if args.seed is not None:
        settings.pipeline["generation"]["seed"] = args.seed

    print(f"[1/5] Loading source PDF: {settings.pdf_path}")
    doc = load_source(settings.pdf_path)
    save_parsed_text(doc, SOURCE_DIR)
    print(f"      sha256={doc.sha256[:12]}…  pages={doc.num_pages}")

    print("[2/5] Building clause index…")
    index = build_clause_index(doc)
    print(f"      {len(index)} clauses, {len(index.definitions)} definitions")

    print("[3/5] Generating records via DeepEval Synthesizer (Anthropic Claude)…")
    from .generation import generate

    records = generate(index, settings)
    print(f"      {len(records)} raw records")

    print("[4/5] Post-processing (citation verify + dedup + validate)…")
    records, report = postprocess(records, index)
    print(
        f"      kept={report.kept}  dropped_citation={len(report.dropped_citation)}  "
        f"dropped_duplicate={len(report.dropped_duplicate)}  "
        f"dropped_language={len(report.dropped_language)}"
    )

    print("[5/5] Curating + freezing dataset…")
    result = curate(records, settings, PROVIDER, report)
    print(f"      dataset:  {result['dataset_path']}")
    print(f"      card:     {result['card_path']}")
    print(f"      manifest: {result['manifest_path']}")
    print("Done.")
    return 0


def _cmd_evaluate(args) -> int:
    from pathlib import Path

    from .evaluation import evaluate

    settings = load_settings()
    dataset = Path(args.dataset) if args.dataset else (DATASET_DIR / "dataset.jsonl")
    if not dataset.exists():
        print(f"Dataset not found: {dataset}. Run `generate` first.", file=sys.stderr)
        return 1
    print(f"Evaluating {dataset} (judge=Anthropic Claude)…")
    out = evaluate(settings, dataset_path=dataset)
    print(f"Wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="razorpay-qa",
        description="Synthetic Q&A dataset pipeline for a Razorpay ToS compliance assistant.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Generate, verify, curate and freeze the dataset.")
    g.add_argument("--seed", type=int, default=None, help="Override the RNG seed.")
    g.set_defaults(func=_cmd_generate)

    e = sub.add_parser("evaluate", help="Run the LLM-as-judge evaluation + write summary.")
    e.add_argument("--dataset", default=None, help="Path to dataset.jsonl (defaults to artifacts/dataset).")
    e.set_defaults(func=_cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    _load_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
