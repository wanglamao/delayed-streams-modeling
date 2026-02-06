#!/usr/bin/env python3
"""
测试 LoRA adapter checkpoint 推理效果

用法示例:
  python scripts/stt_test_lora_checkpoint.py \
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
#     "pyyaml",
# ]
# ///

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import julius
import moshi.models
import moshi.modules.lora
import safetensors.torch
import sphn
import torch


def load_lora_checkpoint(checkpoint_dir: Path, device: str = "cuda"):
    """加载 LoRA adapter checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)

    # 构建路径 - 检查是否需要进入 consolidated 子目录
    if (checkpoint_dir / "consolidated").exists():
        checkpoint_dir = checkpoint_dir / "consolidated"

    config_path = checkpoint_dir / "config.json"
    lora_weight = checkpoint_dir / "lora.safetensors"

    if not config_path.exists():
        print(f"❌ 找不到配置文件: {config_path}")
        sys.exit(1)
    if not lora_weight.exists():
        print(f"❌ 找不到 LoRA 权重: {lora_weight}")
        sys.exit(1)

    # 读取配置
    with open(config_path) as f:
        config = json.load(f)

    # 尝试从 run_dir 的 args.yaml 读取基座模型路径
    run_dir = checkpoint_dir.parent.parent.parent
    args_path = run_dir / "args.yaml"

    import yaml
    base_model_path = None
    mimi_path = None
    tokenizer_path = None

    if args_path.exists():
        with open(args_path) as f:
            args = yaml.safe_load(f)
            moshi_paths = args.get("moshi_paths", {})
            base_model_path = moshi_paths.get("moshi_path")
            mimi_path = moshi_paths.get("mimi_path")
            tokenizer_path = moshi_paths.get("tokenizer_path")

    if not base_model_path or not Path(base_model_path).exists():
        print(f"❌ 无法找到基座模型路径")
        print(f"   检查了: {args_path}")
        print(f"   路径: {base_model_path}")
        sys.exit(1)

    # 获取 LoRA 配置
    lora_rank = config.get("lora_rank", 64)
    lora_scaling = config.get("lora_scaling", 2.0)

    print(f"[加载] LoRA Checkpoint")
    print(f"  Config: {config_path}")
    print(f"  LoRA: {lora_weight}")
    print(f"  Base Model: {base_model_path}")
    print(f"  Mimi: {mimi_path}")
    print(f"  Tokenizer: {tokenizer_path}")
    print(f"  LoRA rank: {lora_rank}, scaling: {lora_scaling}")

    # 创建 CheckpointInfo，传入 lora_weights
    info = moshi.models.loaders.CheckpointInfo.from_hf_repo(
        "kyutai/stt-1b-en_fr",  # dummy repo
        moshi_weights=str(base_model_path),
        mimi_weights=str(mimi_path),
        tokenizer=str(tokenizer_path),
        config_path=str(config_path),
        lora_weights=str(lora_weight),  # 关键：传入 LoRA 权重路径
    )

    print(f"[加载] Mimi 音频编解码器")
    mimi = info.get_mimi(device=device)

    print(f"[加载] Tokenizer")
    tokenizer = info.get_text_tokenizer()

    print(f"[加载] 语言模型 (含 LoRA adapter)")
    # info.get_moshi() 会自动处理 LoRA 转换和权重加载
    lm = info.get_moshi(
        device=device,
        dtype=torch.bfloat16,
    )

    # 设置为评估模式
    lm.eval()

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
):
    """转录单个音频文件"""

    # 加载配置
    audio_silence_prefix_seconds = info.stt_config.get("audio_silence_prefix_seconds", 0.0)
    audio_delay_seconds = info.stt_config.get("audio_delay_seconds", 0.5)

    # 加载音频
    audio, input_sample_rate = sphn.read(str(audio_path))
    audio = torch.from_numpy(audio).to(device)
    audio = julius.resample_frac(audio, input_sample_rate, mimi.sample_rate)

    # Padding 对齐
    if audio.shape[-1] % mimi.frame_size != 0:
        to_pad = mimi.frame_size - audio.shape[-1] % mimi.frame_size
        audio = torch.nn.functional.pad(audio, (0, to_pad))

    duration = audio.shape[-1] / mimi.sample_rate

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

    print(f"\n{'='*60}")
    print(f"开始推理...")
    print(f"{'='*60}\n")

    # 推理
    with mimi.streaming(1), lm_gen.streaming(1):
        for audio_chunk in chunks:
            audio_tokens = mimi.encode(audio_chunk)
            text_tokens = lm_gen.step(audio_tokens)
            text_tokens_accum.append(text_tokens)

            # 实时打印
            text_token = text_tokens[0, 0, 0].cpu().item()
            if text_token not in (0, 3):
                _text = tokenizer.id_to_piece(text_token)  # type: ignore
                _text = _text.replace("▁", " ")
                print(_text, end="", flush=True)

    print(f"\n\n{'='*60}")
    print(f"推理完成")
    print(f"{'='*60}")

    # 解码
    utterance_tokens = torch.concat(text_tokens_accum, dim=-1)
    all_token_ids = utterance_tokens[0, 0].cpu().tolist()
    filtered_token_ids = [t for t in all_token_ids if t not in (0, 3)]
    decoded_text = tokenizer.decode(filtered_token_ids)

    print(f"\n[转录结果]")
    print(f"  {decoded_text}")

    if verbose:
        print(f"\n[Token 分析]")
        print(f"  Token IDs (前 50 个): {filtered_token_ids[:50]}")
        print(f"  总 Tokens: {len(filtered_token_ids)}")

        # Token 多样性检查
        unique_tokens = len(set(filtered_token_ids))
        print(f"  唯一 Tokens: {unique_tokens}")
        if unique_tokens < 10:
            print(f"  ⚠️  警告: Token 多样性过低 ({unique_tokens} 种)，可能存在 mode collapse！")
        else:
            print(f"  ✓ Token 多样性正常")

    # 读取标准答案
    json_file = audio_path.with_suffix(".json")
    if json_file.exists():
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            alignments = data.get("alignments", [])
            gt_text = "".join([word for word, _, _ in alignments])

            print(f"\n[标准答案]")
            print(f"  {gt_text}")

            # 简单字符级对比
            pred_chars = decoded_text.replace(" ", "")
            gt_chars = gt_text.replace(" ", "")

            if pred_chars == gt_chars:
                print(f"\n✅ 完全正确！")
            else:
                # 简单的编辑距离 CER
                m, n = len(pred_chars), len(gt_chars)
                dp = [[0] * (n + 1) for _ in range(m + 1)]

                for i in range(m + 1):
                    dp[i][0] = i
                for j in range(n + 1):
                    dp[0][j] = j

                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if pred_chars[i - 1] == gt_chars[j - 1]:
                            dp[i][j] = dp[i - 1][j - 1]
                        else:
                            dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

                cer = dp[m][n] / n if n > 0 else 0

                print(f"\n[字符错误率 (CER)]")
                print(f"  {cer*100:.2f}%")

                if cer < 0.05:
                    print(f"  ✓ 非常好")
                elif cer < 0.15:
                    print(f"  ⚠️  有少量错误")
                else:
                    print(f"  ❌ 错误较多")


def main():
    parser = argparse.ArgumentParser(
        description="测试 LoRA adapter checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="LoRA checkpoint 目录 (包含 lora.safetensors 和 config.json)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="音频文件路径",
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

    args = parser.parse_args()

    # 加载模型
    checkpoint_dir = Path(args.checkpoint)
    mimi, tokenizer, lm_gen, info = load_lora_checkpoint(checkpoint_dir, device=args.device)

    # 转录
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"❌ 找不到音频文件: {audio_path}")
        sys.exit(1)

    transcribe_audio(audio_path, mimi, tokenizer, lm_gen, info, device=args.device, verbose=args.verbose)


if __name__ == "__main__":
    main()
