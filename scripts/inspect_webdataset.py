# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "webdataset",
#     "soundfile",
#     "numpy",
#     "tqdm",
# ]
# ///
"""Inspect and iterate over WebDataset tar files.

Examples:
    # List all shards and basic statistics
    python scripts/inspect_webdataset.py --data_dir data/stt_zh_webdataset

    # Show detailed sample information
    python scripts/inspect_webdataset.py --data_dir data/stt_zh_webdataset --show_samples 5

    # Verify all samples can be decoded
    python scripts/inspect_webdataset.py --data_dir data/stt_zh_webdataset --verify

    # Export sample keys to file
    python scripts/inspect_webdataset.py --data_dir data/stt_zh_webdataset --export_keys keys.txt

    # Show audio statistics
    python scripts/inspect_webdataset.py --data_dir data/stt_zh_webdataset --audio_stats
"""

import json
import io
from pathlib import Path
from typing import Optional
import argparse

import webdataset as wds
import soundfile as sf
import numpy as np
from tqdm import tqdm


def list_shards(data_dir: str) -> list[Path]:
    """List all tar files in directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Directory not found: {data_dir}")
    shards = sorted(data_path.glob("*.tar"))
    return shards


def inspect_webdataset(
    data_dir: str,
    show_samples: int = 0,
    verify: bool = False,
    export_keys: Optional[str] = None,
    audio_stats: bool = False,
):
    """Inspect WebDataset and show statistics."""

    shards = list_shards(data_dir)
    print(f"=" * 70)
    print(f"WebDataset Inspector")
    print(f"=" * 70)
    print(f"\nData directory: {data_dir}")
    print(f"Number of shards: {len(shards)}")

    if not shards:
        print("No .tar files found!")
        return

    # Show shard files
    print(f"\nShard files:")
    total_size = 0
    for shard in shards:
        size_mb = shard.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  {shard.name:20s} {size_mb:8.2f} MB")
    print(f"  {'Total':20s} {total_size:8.2f} MB")

    # Build dataset pipeline
    urls = [str(s) for s in shards]
    dataset = wds.DataPipeline(
        wds.SimpleShardList(urls),
        wds.tarfile_to_samples(),
        wds.decode(),
    )

    # Collect statistics
    total_samples = 0
    total_duration = 0.0
    keys = []
    errors = []

    audio_durations = []
    audio_sample_rates = set()

    print(f"\nScanning samples...")
    for sample in tqdm(dataset, desc="Processing"):
        total_samples += 1
        key = sample.get("__key__", f"sample_{total_samples}")
        keys.append(key)

        try:
            # Parse metadata
            metadata = sample["json"]
            if isinstance(metadata, bytes):
                metadata = json.loads(metadata)
            print(f"\n[{total_samples}] Key: {key}")
            print(f"  Path: {metadata.get('path', 'N/A')}")
            print(f"  Duration: {metadata.get('duration', 'N/A'):.2f}s")
            print(f"  Text preview: {metadata.get('text', 'N/A')[:60]}...")
            alignments = metadata.get('alignments', [])
            print(f"  Alignments count: {len(alignments)}")
            duration = metadata.get("duration", 0)
            total_duration += duration

            # Verify audio
            if verify or audio_stats:
                audio_bytes = sample["wav"]
                audio_data, sr = sf.read(io.BytesIO(audio_bytes))
                audio_sample_rates.add(sr)

                if audio_stats:
                    audio_duration = len(audio_data) / sr
                    audio_durations.append(audio_duration)

        except Exception as e:
            errors.append((key, str(e)))

    # Print statistics
    print(f"\n{'=' * 70}")
    print(f"Dataset Statistics")
    print(f"{'=' * 70}")
    print(f"Total samples: {total_samples}")
    print(f"Total audio duration: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours)")
    print(f"Average duration per sample: {total_duration/total_samples:.2f} seconds" if total_samples > 0 else "N/A")

    if audio_stats and audio_durations:
        print(f"\nAudio Statistics:")
        print(f"  Sample rates: {audio_sample_rates}")
        print(f"  Min duration: {min(audio_durations):.2f}s")
        print(f"  Max duration: {max(audio_durations):.2f}s")
        print(f"  Mean duration: {np.mean(audio_durations):.2f}s")
        print(f"  Median duration: {np.median(audio_durations):.2f}s")

    if errors:
        print(f"\n⚠️  Errors ({len(errors)}):")
        for key, error in errors[:10]:
            print(f"  {key}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    # Show sample details
    if show_samples > 0:
        print(f"\n{'=' * 70}")
        print(f"Sample Details (first {show_samples})")
        print(f"{'=' * 70}")

        dataset = wds.DataPipeline(
            wds.SimpleShardList(urls),
            wds.tarfile_to_samples(),
            wds.decode(),
        )

        for i, sample in enumerate(dataset):
            if i >= show_samples:
                break

            print(f"\n[{i+1}] Key: {sample.get('__key__', 'N/A')}")

            metadata = sample["json"]
            if isinstance(metadata, bytes):
                metadata = json.loads(metadata)

            print(f"  Path: {metadata.get('path', 'N/A')}")
            print(f"  Duration: {metadata.get('duration', 'N/A'):.2f}s")
            print(f"  Text preview: {metadata.get('text', 'N/A')[:60]}...")

            alignments = metadata.get('alignments', [])
            print(f"  Alignments count: {len(alignments)}")

            audio_bytes = sample["wav"]
            audio_data, sr = sf.read(io.BytesIO(audio_bytes))
            print(f"  Audio shape: {audio_data.shape}")
            print(f"  Sample rate: {sr}")

    # Export keys
    if export_keys:
        with open(export_keys, "w") as f:
            for key in keys:
                f.write(f"{key}\n")
        print(f"\n✓ Exported {len(keys)} keys to: {export_keys}")

    print(f"\n{'=' * 70}")


def iterate_webdataset(
    data_dir: str,
    pattern: Optional[str] = None,
):
    """Iterate over WebDataset samples (generator).

    Args:
        data_dir: Directory containing tar files, or pattern like "shard-{000000..000999}.tar"
        pattern: Optional brace expansion pattern

    Yields:
        dict with keys: key, json, wav, audio_data, sample_rate

    Example:
        for sample in iterate_webdataset("data/stt_zh_webdataset"):
            print(sample["key"], sample["json"]["duration"])
            audio = sample["audio_data"]  # numpy array
    """
    if pattern:
        urls = str(Path(data_dir) / pattern)
    else:
        shards = list_shards(data_dir)
        urls = [str(s) for s in shards]

    dataset = wds.DataPipeline(
        wds.SimpleShardList(urls),
        wds.tarfile_to_samples(),
        wds.decode(),
    )

    for sample in dataset:
        metadata = sample["json"]
        if isinstance(metadata, bytes):
            metadata = json.loads(metadata)

        audio_bytes = sample["wav"]
        audio_data, sr = sf.read(io.BytesIO(audio_bytes))

        yield {
            "key": sample.get("__key__"),
            "json": metadata,
            "wav": audio_bytes,
            "audio_data": audio_data,
            "sample_rate": sr,
        }


def main():
    parser = argparse.ArgumentParser(description="Inspect WebDataset tar files")
    parser.add_argument("--data_dir", required=True, help="Directory containing tar files")
    parser.add_argument("--show_samples", type=int, default=0, help="Show N sample details")
    parser.add_argument("--verify", action="store_true", help="Verify all samples can be decoded")
    parser.add_argument("--export_keys", type=str, help="Export sample keys to file")
    parser.add_argument("--audio_stats", action="store_true", help="Collect audio statistics")

    args = parser.parse_args()

    inspect_webdataset(
        data_dir=args.data_dir,
        show_samples=args.show_samples,
        verify=args.verify,
        export_keys=args.export_keys,
        audio_stats=args.audio_stats,
    )


if __name__ == "__main__":
    main()
