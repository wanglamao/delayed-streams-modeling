#!/usr/bin/env python3
"""
测试微调后的 STT checkpoint 推理效果

用法示例:
  # 测试单个音频文件
  python scripts/stt_test_checkpoint.py \
    --checkpoint /path/to/checkpoint/consolidated \
    --audio test.wav

  # 批量测试多个文件
  python scripts/stt_test_checkpoint.py \
    --checkpoint /path/to/checkpoint/consolidated \
    --audio data/stt_zh/wavs/*.wav \
    --batch

  # 详细的 token 分析
  python scripts/stt_test_checkpoint.py \
    --checkpoint /path/to/checkpoint/consolidated \
    --audio test.wav \
    --verbose
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "moshi>=0.2.11",
#     "torch",
#     "julius",
#     "sphn",
#     "safetensors",
# ]
# ///

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple

import julius
import moshi.models
import sphn
import torch


def load_checkpoint(checkpoint_dir: Path, device: str = "cuda"):
    """加载 STT checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)

    # 构建路径
    config_path = checkpoint_dir / "config.json"
    moshi_weight = checkpoint_dir / "consolidated.safetensors"

    # 查找 mimi 和 tokenizer (通常在父目录或固定位置)
    # 优先使用 checkpoint 同级目录
    parent = checkpoint_dir.parent.parent.parent
    mimi_weight = parent / "mimi.safetensors"
    tokenizer_path = parent / "tokenizer_spm_32k_3.model"

    # 如果找不到，尝试从环境变量或默认位置
    if not mimi_weight.exists():
        # 尝试 run_dir 根目录
        run_root = checkpoint_dir.parent.parent.parent
        mimi_weight = run_root / "mimi.safetensors"

    if not tokenizer_path.exists():
        run_root = checkpoint_dir.parent.parent.parent
        tokenizer_path = run_root / "tokenizer_spm_32k_3.model"

    if not config_path.exists():
        print(f"❌ 找不到配置文件: {config_path}")
        sys.exit(1)
    if not moshi_weight.exists():
        print(f"❌ 找不到模型权重: {moshi_weight}")
        sys.exit(1)
    if not mimi_weight.exists():
        print(f"❌ 找不到 Mimi 权重: {mimi_weight}")
        print(f"   请手动指定: --mimi-weight /path/to/mimi.safetensors")
        sys.exit(1)
    if not tokenizer_path.exists():
        print(f"❌ 找不到 tokenizer: {tokenizer_path}")
        print(f"   请手动指定: --tokenizer /path/to/tokenizer.model")
        sys.exit(1)

    print(f"[加载] Checkpoint 路径")
    print(f"  Config: {config_path}")
    print(f"  Moshi: {moshi_weight}")
    print(f"  Mimi: {mimi_weight}")
    print(f"  Tokenizer: {tokenizer_path}")

    info = moshi.models.loaders.CheckpointInfo.from_hf_repo(
        "kyutai/stt-1b-en_fr",  # dummy repo
        moshi_weights=str(moshi_weight),
        mimi_weights=str(mimi_weight),
        tokenizer=str(tokenizer_path),
        config_path=str(config_path),
    )

    print(f"[加载] Mimi 音频编解码器")
    mimi = info.get_mimi(device=device)

    print(f"[加载] Tokenizer")
    tokenizer = info.get_text_tokenizer()

    print(f"[加载] 语言模型")
    lm = info.get_moshi(device=device, dtype=torch.bfloat16)
    lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)

    return mimi, tokenizer, lm_gen, info


def transcribe_audio(
    audio_path: Path,
    mimi,
    tokenizer,
    lm_gen,
    info,
    device: str = "cuda",
    verbose: bool = False,
) -> Tuple[str, List[int], float]:
    """转录单个音频文件"""

    # 加载配置
    audio_silence_prefix_seconds = info.stt_config.get("audio_silence_prefix_seconds", 0.0)
    audio_delay_seconds = info.stt_config.get("audio_delay_seconds", 0.5)
    padding_token_id = info.raw_config.get("text_padding_token_id", 3)

    # 加载音频
    audio, input_sample_rate = sphn.read(str(audio_path))
    audio = torch.from_numpy(audio).to(device)
    audio = julius.resample_frac(audio, input_sample_rate, mimi.sample_rate)

    # Padding 对齐
    if audio.shape[-1] % mimi.frame_size != 0:
        to_pad = mimi.frame_size - audio.shape[-1] % mimi.frame_size
        audio = torch.nn.functional.pad(audio, (0, to_pad))

    duration = audio.shape[-1] / mimi.sample_rate

    if verbose:
        print(f"\n[音频信息]")
        print(f"  文件: {audio_path.name}")
        print(f"  时长: {duration:.2f}s")
        print(f"  Audio delay: {audio_delay_seconds}s")
        print(f"  Silence prefix: {audio_silence_prefix_seconds}s")

    # 准备音频 chunks
    text_tokens_accum = []
    n_prefix_chunks = math.ceil(audio_silence_prefix_seconds * mimi.frame_rate)
    n_suffix_chunks = math.ceil(audio_delay_seconds * mimi.frame_rate)
    silence_chunk = torch.zeros((1, 1, mimi.frame_size), dtype=torch.float32, device=device)

    chunks = itertools.chain(
        itertools.repeat(silence_chunk, n_prefix_chunks),
        torch.split(audio[:, None], mimi.frame_size, dim=-1),
        itertools.repeat(silence_chunk, n_suffix_chunks),
    )

    # 推理
    with mimi.streaming(1), lm_gen.streaming(1):
        for audio_chunk in chunks:
            audio_tokens = mimi.encode(audio_chunk)
            text_tokens = lm_gen.step(audio_tokens)
            text_tokens_accum.append(text_tokens)

            # 实时打印（verbose 模式）
            if verbose:
                text_token = text_tokens[0, 0, 0].cpu().item()
                if text_token not in (0, 3):
                    _text = tokenizer.id_to_piece(text_token)  # type: ignore
                    _text = _text.replace("▁", " ")
                    print(_text, end="", flush=True)

    if verbose:
        print()  # newline

    # 解码
    utterance_tokens = torch.concat(text_tokens_accum, dim=-1)
    all_token_ids = utterance_tokens[0, 0].cpu().tolist()
    filtered_token_ids = [t for t in all_token_ids if t not in (0, 3)]
    decoded_text = tokenizer.decode(filtered_token_ids)

    return decoded_text, filtered_token_ids, duration


def load_ground_truth(audio_path: Path) -> str:
    """加载标准答案（从对应的 .json 文件）"""
    json_file = audio_path.with_suffix(".json")
    if json_file.exists():
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            alignments = data.get("alignments", [])
            return "".join([word for word, _, _ in alignments])
    return ""


def calculate_cer(pred: str, gt: str) -> float:
    """简单的字符错误率计算 (Levenshtein distance)"""
    pred = pred.replace(" ", "")
    gt = gt.replace(" ", "")

    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0

    # 简单的编辑距离
    m, n = len(pred), len(gt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == gt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    return dp[m][n] / n


def main():
    parser = argparse.ArgumentParser(
        description="测试微调后的 STT checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint 目录 (包含 consolidated.safetensors 和 config.json)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        nargs="+",
        required=True,
        help="音频文件路径（支持通配符，如 *.wav）",
    )
    parser.add_argument(
        "--mimi-weight",
        type=str,
        help="Mimi 权重路径 (可选，默认自动查找)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Tokenizer 路径 (可选，默认自动查找)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="运行设备 (默认: cuda)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细信息（token 分析等）",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="批量测试模式（显示汇总统计）",
    )

    args = parser.parse_args()

    # 加载模型
    checkpoint_dir = Path(args.checkpoint)
    mimi, tokenizer, lm_gen, info = load_checkpoint(checkpoint_dir, device=args.device)

    # 处理音频文件列表
    audio_files = []
    for pattern in args.audio:
        audio_files.extend(Path(".").glob(pattern))

    if not audio_files:
        print(f"❌ 找不到音频文件: {args.audio}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"开始推理 ({len(audio_files)} 个文件)")
    print(f"{'='*70}\n")

    # 批量测试
    results = []
    for i, audio_path in enumerate(audio_files, 1):
        if args.batch:
            print(f"[{i}/{len(audio_files)}] {audio_path.name}...", end=" ", flush=True)
        else:
            print(f"\n{'='*70}")
            print(f"[{i}/{len(audio_files)}] {audio_path}")
            print(f"{'='*70}")

        # 转录
        decoded_text, token_ids, duration = transcribe_audio(
            audio_path, mimi, tokenizer, lm_gen, info, device=args.device, verbose=args.verbose and not args.batch
        )

        # 加载标准答案
        gt_text = load_ground_truth(audio_path)

        # 计算 CER
        cer = calculate_cer(decoded_text, gt_text) if gt_text else None

        results.append({
            "file": audio_path.name,
            "pred": decoded_text,
            "gt": gt_text,
            "cer": cer,
            "duration": duration,
            "token_ids": token_ids,
        })

        # 打印结果
        if args.batch:
            if cer is not None:
                print(f"CER: {cer*100:.1f}%")
            else:
                print("✓")
        else:
            print(f"\n[转录结果]")
            print(f"  {decoded_text}")

            if gt_text:
                print(f"\n[标准答案]")
                print(f"  {gt_text}")
                print(f"\n[字符错误率 (CER)]")
                print(f"  {cer*100:.2f}%")

                if cer == 0:
                    print(f"  ✅ 完全正确！")
                elif cer < 0.05:
                    print(f"  ✓ 非常好")
                elif cer < 0.15:
                    print(f"  ⚠️  有少量错误")
                else:
                    print(f"  ❌ 错误较多")

            if args.verbose:
                print(f"\n[Token 分析]")
                print(f"  Token IDs (前 50 个): {token_ids[:50]}")
                print(f"  总 Tokens: {len(token_ids)}")

                # Token 多样性检查
                unique_tokens = len(set(token_ids))
                print(f"  唯一 Tokens: {unique_tokens}")
                if unique_tokens < 10:
                    print(f"  ⚠️  警告: Token 多样性过低，可能存在 mode collapse！")

    # 汇总统计
    if args.batch and len(results) > 1:
        print(f"\n{'='*70}")
        print(f"汇总统计")
        print(f"{'='*70}")

        total_duration = sum(r["duration"] for r in results)
        avg_cer = sum(r["cer"] for r in results if r["cer"] is not None) / len([r for r in results if r["cer"] is not None])

        print(f"总文件数: {len(results)}")
        print(f"总时长: {total_duration:.2f}s ({total_duration/60:.2f}min)")
        if any(r["cer"] is not None for r in results):
            print(f"平均 CER: {avg_cer*100:.2f}%")

        # Token 多样性检查
        all_tokens = []
        for r in results:
            all_tokens.extend(r["token_ids"])
        unique_tokens = len(set(all_tokens))
        print(f"唯一 Tokens: {unique_tokens} / {len(all_tokens)} ({unique_tokens/len(all_tokens)*100:.1f}%)")

        if unique_tokens < 100:
            print(f"\n⚠️  警告: 整体 Token 多样性过低 ({unique_tokens} 种)，可能存在 mode collapse！")


if __name__ == "__main__":
    main()
