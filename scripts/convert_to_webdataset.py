#!/usr/bin/env python3
"""
Convert STT training data to WebDataset format for efficient handling of billions of samples.

WebDataset uses tar archives with a specific naming convention:
- sample_000000.wav (audio file)
- sample_000000.json (metadata: alignments, duration, etc.)

Usage:
    # Using conda environment (recommended in this project)
    conda activate ala
    python scripts/convert_to_webdataset.py \
        --input_dir data/stt_zh \
        --output_dir data/stt_zh_webdataset \
        --samples_per_shard 1000

For large datasets, this script will create multiple tar shards (e.g., shard-000000.tar, shard-000001.tar, ...)
to keep individual archives at a manageable size.

Recommended shard sizes:
- Small datasets (< 100k samples): samples_per_shard=1000
- Medium datasets (100k-10M): samples_per_shard=5000
- Large datasets (> 10M): samples_per_shard=10000

See WEBDATASET_GUIDE.md for more details.
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "webdataset>=0.2.86",
#     "fire>=0.7.1",
# ]
# ///

import json
import logging
from pathlib import Path

import fire
import webdataset as wds

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def convert_to_webdataset(
    input_dir: str,
    output_dir: str,
    samples_per_shard: int = 1000,
    pattern: str = "shard-%06d.tar",
):
    """
    Convert training data to WebDataset format.

    Args:
        input_dir: Directory containing train.jsonl and wavs/
        output_dir: Output directory for webdataset shards
        samples_per_shard: Number of samples per tar file (default: 1000)
        pattern: Naming pattern for shards (default: shard-%06d.tar)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all jsonl files
    jsonl_files = sorted(input_path.rglob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No .jsonl files found in {input_dir}")

    logger.info(f"Found {len(jsonl_files)} jsonl file(s)")

    # Process each jsonl file
    for jsonl_file in jsonl_files:
        logger.info(f"Processing {jsonl_file}")

        # Determine output pattern for this jsonl file
        if len(jsonl_files) > 1:
            # Multiple jsonl files: create subdirectories
            relative_path = jsonl_file.relative_to(input_path).parent
            out_dir = output_path / relative_path / jsonl_file.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            shard_pattern = str(out_dir / pattern)
        else:
            # Single jsonl file: output directly to output_dir
            shard_pattern = str(output_path / pattern)

        # Create WebDataset writer with sharding
        with wds.ShardWriter(shard_pattern, maxcount=samples_per_shard) as sink:
            sample_count = 0

            with open(jsonl_file) as f:
                for idx, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        if not data:
                            continue

                        # Get audio path (relative to jsonl file)
                        audio_path = jsonl_file.parent / data["path"]
                        json_path = audio_path.with_suffix(".json")

                        if not audio_path.exists():
                            logger.warning(f"Audio file not found: {audio_path}")
                            continue

                        # Read audio file
                        with open(audio_path, "rb") as audio_f:
                            audio_data = audio_f.read()

                        # Read metadata json if exists
                        metadata = {"duration": data["duration"]}
                        if json_path.exists():
                            with open(json_path) as json_f:
                                metadata.update(json.load(json_f))

                        # Create webdataset sample
                        # Key format: unique identifier for each sample
                        key = f"{jsonl_file.stem}_{idx:08d}"

                        sample = {
                            "__key__": key,
                            "wav": audio_data,  # audio bytes
                            "json": json.dumps(metadata).encode(
                                "utf-8"
                            ),  # metadata as json string
                        }

                        sink.write(sample)
                        sample_count += 1

                        if (sample_count % 1000) == 0:
                            logger.info(f"Processed {sample_count} samples...")

                    except Exception as e:
                        logger.error(f"Error processing line {idx}: {e}")
                        continue

            logger.info(
                f"Completed {jsonl_file}: {sample_count} samples written to {shard_pattern}"
            )

    logger.info("Conversion complete!")

    # Write a manifest file
    manifest_path = output_path / "manifest.json"
    manifest = {
        "input_dir": str(input_path),
        "samples_per_shard": samples_per_shard,
        "pattern": pattern,
        "jsonl_files": [str(f.relative_to(input_path)) for f in jsonl_files],
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    fire.Fire(convert_to_webdataset)
