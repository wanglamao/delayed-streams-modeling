#!/usr/bin/env python3
"""
手动评测 STT checkpoint 在 eval 数据集上的效果 (支持 LoRA 和完整 checkpoint)

用法示例:
  # 评测 LoRA checkpoint
  python scripts/stt_eval_checkpoint.py \
    --checkpoint /path/to/run_dir/checkpoints/checkpoint_000500/consolidated \
    --eval-data /path/to/eval_webdataset

  # 评测完整 checkpoint
  python scripts/stt_eval_checkpoint.py \
    --checkpoint /path/to/checkpoint/consolidated \
    --eval-data /path/to/eval_webdataset \
    --max-samples 100

  # 指定基座模型路径 (LoRA 模式)
  python scripts/stt_eval_checkpoint.py \
    --checkpoint /path/to/checkpoint/consolidated \
    --eval-data /path/to/eval_webdataset \
    --base-model /path/to/base_model.safetensors \
    --mimi-weight /path/to/mimi.safetensors \
    --tokenizer /path/to/tokenizer.model
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
#     "webdataset",
#     "scipy",
#     "tqdm",
# ]
# ///

import argparse
import io
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Iterator, Tuple

import julius
import moshi.models
import numpy as np
import sphn
import torch
import webdataset as wds
from scipy.io import wavfile
from tqdm import tqdm


def load_checkpoint(
    checkpoint_dir: Path,
    device: str = "cuda",
    base_model_path: str = None,
    mimi_weight_path: str = None,
    tokenizer_path: str = None,
):
    """加载 STT checkpoint (自动检测 LoRA 或普通模型)"""
    checkpoint_dir = Path(checkpoint_dir)

    # 检查是否需要进入 consolidated 子目录
    if (checkpoint_dir / "consolidated").exists():
        checkpoint_dir = checkpoint_dir / "consolidated"

    # 构建路径
    config_path = checkpoint_dir / "config.json"

    # 检测 checkpoint 类型
    is_lora = (checkpoint_dir / "lora.safetensors").exists()
    moshi_weight = checkpoint_dir / ("lora.safetensors" if is_lora else "consolidated.safetensors")

    if not config_path.exists():
        print(f"❌ 找不到配置文件: {config_path}")
        sys.exit(1)
    if not moshi_weight.exists():
        print(f"❌ 找不到模型权重: {moshi_weight}")
        sys.exit(1)

    # LoRA checkpoint 需要从 args.yaml 读取基座模型路径
    mimi_weight = mimi_weight_path
    tokenizer = tokenizer_path

    if is_lora:
        # 读取 run_dir 的 args.yaml
        run_dir = checkpoint_dir.parent.parent.parent
        args_path = run_dir / "args.yaml"

        if args_path.exists() and not base_model_path:
            import yaml
            with open(args_path) as f:
                args = yaml.safe_load(f)
                moshi_paths = args.get("moshi_paths", {})
                base_model_path = base_model_path or moshi_paths.get("moshi_path")
                mimi_weight = mimi_weight or moshi_paths.get("mimi_path")
                tokenizer = tokenizer or moshi_paths.get("tokenizer_path")

        if not base_model_path or not Path(base_model_path).exists():
            print(f"❌ LoRA checkpoint 需要基座模型，但找不到路径")
            print(f"   检查了: {args_path}")
            print(f"   请使用 --base-model 指定基座模型路径")
            sys.exit(1)
    else:
        # 普通 checkpoint - 查找 mimi 和 tokenizer
        run_dir = checkpoint_dir.parent.parent.parent

        # 尝试多个位置
        for parent in [run_dir, checkpoint_dir.parent.parent.parent]:
            if not mimi_weight or not Path(str(mimi_weight)).exists():
                candidate = parent / "mimi.safetensors"
                if candidate.exists():
                    mimi_weight = str(candidate)
            if not tokenizer or not Path(str(tokenizer)).exists():
                for name in ["tokenizer.model", "tokenizer_spm_32k_3.model"]:
                    tp = parent / name
                    if tp.exists():
                        tokenizer = str(tp)
                        break

    if not mimi_weight or not Path(str(mimi_weight)).exists():
        print(f"❌ 找不到 Mimi 权重")
        print(f"   请使用: --mimi-weight /path/to/mimi.safetensors")
        sys.exit(1)
    if not tokenizer or not Path(str(tokenizer)).exists():
        print(f"❌ 找不到 tokenizer")
        print(f"   请使用: --tokenizer /path/to/tokenizer.model")
        sys.exit(1)

    # 打印加载信息
    print(f"[加载] {'LoRA' if is_lora else 'Full'} Checkpoint")
    print(f"  Config: {config_path}")
    if is_lora:
        print(f"  Base Model: {base_model_path}")
        print(f"  LoRA Adapter: {moshi_weight}")
    else:
        print(f"  Moshi: {moshi_weight}")
    print(f"  Mimi: {mimi_weight}")
    print(f"  Tokenizer: {tokenizer}")

    # 读取配置
    with open(config_path) as f:
        config = json.load(f)

    if is_lora:
        lora_rank = config.get("lora_rank", 64)
        lora_scaling = config.get("lora_scaling", 2.0)
        print(f"  LoRA rank: {lora_rank}, scaling: {lora_scaling}")

    # 创建 CheckpointInfo
    info = moshi.models.loaders.CheckpointInfo.from_hf_repo(
        "kyutai/stt-1b-en_fr",  # dummy repo
        moshi_weights=str(base_model_path if is_lora else moshi_weight),
        mimi_weights=str(mimi_weight),
        tokenizer=str(tokenizer),
        config_path=str(config_path),
        lora_weights=str(moshi_weight) if is_lora else None,
    )

    print(f"[加载] Mimi 音频编解码器")
    mimi = info.get_mimi(device=device)

    print(f"[加载] Tokenizer")
    text_tokenizer = info.get_text_tokenizer()

    print(f"[加载] 语言模型{'(含 LoRA adapter)' if is_lora else ''}")
    lm = info.get_moshi(device=device, dtype=torch.bfloat16)
    lm.eval()
    lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)

    return mimi, text_tokenizer, lm_gen, info


def decode_audio(data: bytes, target_sample_rate: int) -> np.ndarray:
    """Decode audio bytes to numpy array."""
    bio = io.BytesIO(data)
    sr, audio = wavfile.read(bio)

    # Convert to float32 and normalize
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)

    # Ensure mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=-1)

    return audio, sr


def create_eval_iterator(
    eval_data_path: str,
    max_samples: int = None,
) -> Iterator[dict]:
    """创建 eval 数据迭代器"""
    data_path = Path(eval_data_path)

    if data_path.is_dir():
        tar_files = sorted(data_path.glob("*.tar"))
        if not tar_files:
            raise ValueError(f"No .tar files found in {eval_data_path}")
        urls = [str(f) for f in tar_files]
        print(f"[数据] 找到 {len(urls)} 个 shard 文件")
    else:
        raise ValueError(f"eval_data_path 必须是目录: {eval_data_path}")

    # 创建 webdataset (不做 node splitting，单进程读取所有数据)
    dataset = wds.WebDataset(urls, shardshuffle=False)

    count = 0
    for sample in dataset:
        try:
            # Parse metadata
            json_data = sample["json"]
            if isinstance(json_data, bytes):
                metadata = json.loads(json_data.decode("utf-8"))
            else:
                metadata = json_data

            # Decode audio
            audio, sr = decode_audio(sample["wav"], 24000)

            yield {
                "audio": audio,
                "sample_rate": sr,
                "metadata": metadata,
                "key": sample["__key__"],
                "alignments": metadata.get("alignments", []),
            }

            count += 1
            if max_samples and count >= max_samples:
                break

        except Exception as e:
            print(f"⚠️  跳过样本 {sample.get('__key__', 'unknown')}: {e}")
            continue


def transcribe_audio(
    audio: np.ndarray,
    sample_rate: int,
    mimi,
    tokenizer,
    lm_gen,
    info,
    device: str = "cuda",
) -> Tuple[str, list]:
    """转录音频"""
    audio_silence_prefix_seconds = info.stt_config.get("audio_silence_prefix_seconds", 0.0)
    audio_delay_seconds = info.stt_config.get("audio_delay_seconds", 0.5)

    # 转换为 tensor
    audio_tensor = torch.from_numpy(audio).to(device).unsqueeze(0)  # [1, samples]

    # 重采样
    if sample_rate != mimi.sample_rate:
        audio_tensor = julius.resample_frac(audio_tensor, sample_rate, mimi.sample_rate)

    # Padding 对齐
    if audio_tensor.shape[-1] % mimi.frame_size != 0:
        to_pad = mimi.frame_size - audio_tensor.shape[-1] % mimi.frame_size
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, to_pad))

    # 准备 chunks
    text_tokens_accum = []
    n_prefix_chunks = math.ceil(audio_silence_prefix_seconds * mimi.frame_rate)
    n_suffix_chunks = math.ceil(audio_delay_seconds * mimi.frame_rate)
    silence_chunk = torch.zeros((1, 1, mimi.frame_size), dtype=torch.float32, device=device)

    chunks = itertools.chain(
        itertools.repeat(silence_chunk, n_prefix_chunks),
        torch.split(audio_tensor[:, None], mimi.frame_size, dim=-1),
        itertools.repeat(silence_chunk, n_suffix_chunks),
    )

    # 推理
    with mimi.streaming(1), lm_gen.streaming(1):
        for audio_chunk in chunks:
            audio_tokens = mimi.encode(audio_chunk)
            text_tokens = lm_gen.step(audio_tokens)
            text_tokens_accum.append(text_tokens)

    # 解码
    utterance_tokens = torch.concat(text_tokens_accum, dim=-1)
    all_token_ids = utterance_tokens[0, 0].cpu().tolist()
    filtered_token_ids = [t for t in all_token_ids if t not in (0, 3)]
    decoded_text = tokenizer.decode(filtered_token_ids)

    return decoded_text, filtered_token_ids


def calculate_cer(pred: str, gt: str) -> float:
    """计算字符错误率 (CER)"""
    pred = pred.replace(" ", "").replace("▁", "")
    gt = gt.replace(" ", "").replace("▁", "")

    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0

    # 编辑距离
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
        description="手动评测 STT checkpoint (支持 LoRA 和完整 checkpoint)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint 目录 (自动检测是否为 LoRA)",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        required=True,
        help="Eval 数据目录 (webdataset 格式)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大评测样本数 (默认: 全部)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="基座模型路径 (LoRA 模式需要)",
    )
    parser.add_argument(
        "--mimi-weight",
        type=str,
        default=None,
        help="Mimi 权重路径",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer 路径",
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
        help="显示每个样本的详细结果",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出结果到 JSON 文件",
    )

    args = parser.parse_args()

    # 加载模型
    print("=" * 70)
    print("加载模型")
    print("=" * 70)
    mimi, tokenizer, lm_gen, info = load_checkpoint(
        Path(args.checkpoint),
        device=args.device,
        base_model_path=args.base_model,
        mimi_weight_path=args.mimi_weight,
        tokenizer_path=args.tokenizer,
    )

    # 创建数据迭代器
    print("\n" + "=" * 70)
    print("加载评测数据")
    print("=" * 70)
    eval_iter = create_eval_iterator(args.eval_data, args.max_samples)

    # 评测
    print("\n" + "=" * 70)
    print("开始评测")
    print("=" * 70 + "\n")

    results = []
    total_cer = 0.0
    total_samples = 0
    all_tokens = []

    # 使用 tqdm 显示进度
    for sample in tqdm(eval_iter, desc="评测进度", unit="样本"):
        try:
            # 转录
            pred_text, token_ids = transcribe_audio(
                sample["audio"],
                sample["sample_rate"],
                mimi,
                tokenizer,
                lm_gen,
                info,
                device=args.device,
            )

            # 获取标准答案
            alignments = sample["alignments"]
            gt_text = "".join([word for word, _, _ in alignments])

            # 计算 CER
            cer = calculate_cer(pred_text, gt_text)

            result = {
                "key": sample["key"],
                "pred": pred_text,
                "gt": gt_text,
                "cer": cer,
                "num_tokens": len(token_ids),
            }
            results.append(result)

            total_cer += cer
            total_samples += 1
            all_tokens.extend(token_ids)

            if args.verbose:
                tqdm.write(f"\n[{sample['key']}]")
                tqdm.write(f"  预测: {pred_text[:100]}{'...' if len(pred_text) > 100 else ''}")
                tqdm.write(f"  标签: {gt_text[:100]}{'...' if len(gt_text) > 100 else ''}")
                tqdm.write(f"  CER: {cer*100:.2f}%")

        except Exception as e:
            tqdm.write(f"⚠️  处理样本 {sample['key']} 失败: {e}")
            continue

    # 汇总统计
    print("\n" + "=" * 70)
    print("评测结果汇总")
    print("=" * 70)

    if total_samples == 0:
        print("❌ 没有成功评测任何样本")
        sys.exit(1)

    avg_cer = total_cer / total_samples
    unique_tokens = len(set(all_tokens))

    print(f"总样本数: {total_samples}")
    print(f"平均 CER: {avg_cer*100:.2f}%")
    print(f"总 Token 数: {len(all_tokens)}")
    print(f"唯一 Token 数: {unique_tokens} ({unique_tokens/len(all_tokens)*100:.1f}%)")

    # CER 分布
    cer_values = [r["cer"] for r in results]
    print(f"\nCER 分布:")
    print(f"  最小: {min(cer_values)*100:.2f}%")
    print(f"  最大: {max(cer_values)*100:.2f}%")
    print(f"  中位数: {sorted(cer_values)[len(cer_values)//2]*100:.2f}%")

    # CER 区间统计
    perfect = sum(1 for c in cer_values if c == 0)
    good = sum(1 for c in cer_values if 0 < c <= 0.05)
    medium = sum(1 for c in cer_values if 0.05 < c <= 0.15)
    bad = sum(1 for c in cer_values if c > 0.15)

    print(f"\nCER 区间分布:")
    print(f"  完美 (0%): {perfect} ({perfect/total_samples*100:.1f}%)")
    print(f"  优秀 (0-5%): {good} ({good/total_samples*100:.1f}%)")
    print(f"  一般 (5-15%): {medium} ({medium/total_samples*100:.1f}%)")
    print(f"  较差 (>15%): {bad} ({bad/total_samples*100:.1f}%)")

    # Mode collapse 检测
    if unique_tokens < 100:
        print(f"\n⚠️  警告: Token 多样性过低 ({unique_tokens} 种)，可能存在 mode collapse！")
    elif unique_tokens < 500:
        print(f"\n⚠️  注意: Token 多样性较低 ({unique_tokens} 种)")

    # 输出到文件
    if args.output:
        output_data = {
            "summary": {
                "total_samples": total_samples,
                "avg_cer": avg_cer,
                "unique_tokens": unique_tokens,
                "total_tokens": len(all_tokens),
            },
            "results": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
