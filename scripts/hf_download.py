#!/usr/bin/env python3
"""
Download a Hugging Face repo snapshot into the project directory (no global cache).

Usage (recommended, from repo root):
  cd moshi-finetune
  uv run python ../scripts/hf_download.py kyutai/stt-1b-en_fr-candle

This script forces HF caches under `./.hf_home/` unless you already set `HF_HOME`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _ensure_local_hf_home(project_root: Path) -> None:
    if os.environ.get("HF_HOME"):
        return
    hf_home = project_root / ".hf_home"
    os.environ["HF_HOME"] = str(hf_home)
    # Keep related caches colocated to avoid writing to ~/.cache.
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    os.environ.setdefault("TORCH_HOME", str(hf_home / "torch"))

    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download HF snapshot into ./hf_models/ (project-local)."
    )
    parser.add_argument("repo_id", help="e.g. kyutai/stt-1b-en_fr-candle")
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=("model", "dataset", "space"),
        help="HF repo type",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional git revision/branch/tag/commit on HF Hub",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Where to materialize files (default: ./hf_models/<repo_id>/)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    _ensure_local_hf_home(project_root)

    local_dir = (
        Path(args.local_dir)
        if args.local_dir is not None
        else (project_root / "hf_models" / args.repo_id)
    )
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    snapshot_download(
        args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        cache_dir=os.environ.get("HF_HUB_CACHE"),
        resume_download=True,
    )
    print(f"OK: {args.repo_id} -> {local_dir}")


if __name__ == "__main__":
    main()

