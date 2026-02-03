#!/usr/bin/env python3
"""
Validate a Moshi-Finetune STT dataset manifest (JSONL) + per-wav alignment JSONs.

This is a lightweight checker (stdlib only) so it can run before installing
the full training stack.

Expected:
  - JSONL lines: {"path": "...wav", "duration": 12.34}
  - Sidecar JSON next to wav: {"alignments": [[text, [start, end], speaker], ...]}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Issues:
    errors: int = 0
    warnings: int = 0

    def err(self, msg: str) -> None:
        self.errors += 1
        print(f"[ERROR] {msg}", file=sys.stderr)

    def warn(self, msg: str) -> None:
        self.warnings += 1
        print(f"[WARN ] {msg}", file=sys.stderr)


def wav_info(path: Path) -> tuple[int, int, float]:
    with wave.open(str(path), "rb") as f:
        nchannels = f.getnchannels()
        framerate = f.getframerate()
        nframes = f.getnframes()
    duration = (nframes / framerate) if framerate else 0.0
    return nchannels, framerate, duration


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Moshi-Finetune STT dataset.")
    parser.add_argument("jsonl", type=Path, help="Path to train.jsonl")
    parser.add_argument(
        "--speaker",
        default="SPEAKER_MAIN",
        help="Expected main speaker label (default: SPEAKER_MAIN)",
    )
    parser.add_argument(
        "--require-main-speaker",
        action="store_true",
        help="If set, error when alignment speaker != --speaker",
    )
    parser.add_argument(
        "--duration-tol-sec",
        type=float,
        default=0.5,
        help="Allowed absolute error between manifest duration and wav duration",
    )
    args = parser.parse_args()

    issues = Issues()
    if not args.jsonl.exists():
        issues.err(f"JSONL not found: {args.jsonl}")
        raise SystemExit(2)

    n_items = 0
    total_dur = 0.0

    for line_no, raw_line in enumerate(args.jsonl.read_text(encoding="utf-8").splitlines(), start=1):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError as e:
            issues.err(f"{args.jsonl}:{line_no}: invalid JSON ({e})")
            continue

        path_str = item.get("path")
        dur = item.get("duration")
        if not isinstance(path_str, str) or not path_str:
            issues.err(f"{args.jsonl}:{line_no}: missing/invalid 'path'")
            continue
        if not isinstance(dur, (int, float)) or not math.isfinite(dur) or dur <= 0:
            issues.err(f"{args.jsonl}:{line_no}: missing/invalid 'duration': {dur!r}")
            continue

        wav_path = Path(path_str)
        if not wav_path.is_absolute():
            # Try resolving relative to CWD (common for training runs) then JSONL directory.
            cand_cwd = (Path.cwd() / wav_path).resolve()
            cand_jsonl = (args.jsonl.parent / wav_path).resolve()
            wav_path = cand_cwd if cand_cwd.exists() else cand_jsonl
        if not wav_path.exists():
            issues.err(
                f"{args.jsonl}:{line_no}: wav not found: {wav_path} (path={path_str!r})"
            )
            continue
        if wav_path.suffix.lower() != ".wav":
            issues.warn(f"{args.jsonl}:{line_no}: non-wav file: {wav_path}")

        try:
            nchannels, sr, wav_dur = wav_info(wav_path)
        except wave.Error as e:
            issues.err(f"{args.jsonl}:{line_no}: cannot read wav header ({wav_path}): {e}")
            continue

        if nchannels != 1:
            issues.warn(
                f"{wav_path}: nchannels={nchannels} (recommended mono; training can downmix if enabled)"
            )
        if abs(wav_dur - float(dur)) > args.duration_tol_sec:
            issues.warn(
                f"{wav_path}: duration mismatch (jsonl={dur:.3f}s, wav={wav_dur:.3f}s)"
            )

        sidecar = wav_path.with_suffix(".json")
        if not sidecar.exists():
            issues.err(f"{wav_path}: missing sidecar json: {sidecar}")
            continue
        try:
            side = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            issues.err(f"{sidecar}: invalid JSON ({e})")
            continue

        alignments = side.get("alignments")
        if not isinstance(alignments, list):
            issues.err(f"{sidecar}: missing/invalid 'alignments' list")
            continue

        last_start = -1.0
        for idx, a in enumerate(alignments):
            if not (
                isinstance(a, list)
                and len(a) == 3
                and isinstance(a[0], str)
                and isinstance(a[1], list)
                and len(a[1]) == 2
                and isinstance(a[2], str)
            ):
                issues.err(f"{sidecar}: alignments[{idx}] invalid format: {a!r}")
                continue

            start, end = a[1]
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                issues.err(f"{sidecar}: alignments[{idx}] timestamp not numeric: {a!r}")
                continue
            start_f = float(start)
            end_f = float(end)

            if start_f < 0 or end_f < 0:
                issues.err(f"{sidecar}: alignments[{idx}] negative timestamp: {a!r}")
            if not (start_f < end_f):
                issues.err(f"{sidecar}: alignments[{idx}] requires start < end: {a!r}")
            if start_f < last_start:
                issues.err(
                    f"{sidecar}: alignments not sorted by start (idx={idx}, {start_f:.3f} < {last_start:.3f})"
                )
            last_start = start_f

            if end_f > wav_dur + 0.5:
                issues.warn(
                    f"{sidecar}: alignments[{idx}] end beyond wav duration ({end_f:.3f} > {wav_dur:.3f})"
                )

            if args.require_main_speaker and a[2] != args.speaker:
                issues.err(
                    f"{sidecar}: alignments[{idx}] speaker={a[2]!r} != {args.speaker!r}"
                )

        n_items += 1
        total_dur += float(dur)

    print(f"Checked {n_items} items, total duration ~ {total_dur/3600:.2f} h")
    if issues.errors:
        print(f"FAILED: {issues.errors} errors, {issues.warnings} warnings", file=sys.stderr)
        raise SystemExit(1)
    print(f"OK: {issues.warnings} warnings")


if __name__ == "__main__":
    main()
