#!/usr/bin/env python3
"""
测试 STT checkpoint 推理
"""
import sys
import torch
import sentencepiece
import sphn
from pathlib import Path
from moshi.models import loaders
from moshi.run_inference import InferenceState

def main():
    # 路径配置
    moshi_weight = "/nvmedata/ala_storage/ala_runs/stt_zh_filtered_2_5m_run2/checkpoints/checkpoint_001000/consolidated/consolidated.safetensors"
    config_path = "/nvmedata/ala_storage/ala_runs/stt_zh_filtered_2_5m_run2/checkpoints/checkpoint_001000/consolidated/config.json"
    mimi_weight = "/nvmedata/ala_storage/stt-1b-en_fr-candle/mimi-pytorch-e351c8d8@125.safetensors"
    tokenizer_path = "/nvmedata/ala_storage/stt-1b-en_fr-candle/tokenizer_en_fr_audio_8000.model"
    audio_file = "data/stt_zh/wavs/emilia_zh_0000983244.wav"

    device = "cuda"
    batch_size = 1

    print(f"[Info] 加载 checkpoint 配置")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo="kyutai/moshiko-pytorch-bf16",  # 仅用于默认值
        moshi_weights=moshi_weight,
        mimi_weights=mimi_weight,
        tokenizer=tokenizer_path,
        config_path=config_path,
    )

    print(f"[Info] 加载 Mimi (音频编解码器)")
    mimi = checkpoint_info.get_mimi(device=device)

    print(f"[Info] 加载 Tokenizer")
    text_tokenizer = checkpoint_info.get_text_tokenizer()

    print(f"[Info] 加载 Moshi (语言模型)")
    lm = checkpoint_info.get_moshi(device=device, dtype=torch.bfloat16)

    print(f"[Info] 加载音频文件: {audio_file}")
    in_pcms, _ = sphn.read(audio_file, sample_rate=mimi.sample_rate)
    in_pcms = torch.from_numpy(in_pcms).to(device=device)
    in_pcms = in_pcms[None, 0:1].expand(batch_size, -1, -1)

    print(f"[Info] 音频长度: {in_pcms.shape[-1] / mimi.sample_rate:.2f} 秒")

    # 初始化推理状态
    state = InferenceState(
        checkpoint_info,
        mimi,
        text_tokenizer,
        lm,
        batch_size,
        cfg_coef=1.0,
        device=device,
        **checkpoint_info.lm_gen_config,
    )

    print(f"\n{'='*60}")
    print(f"开始推理...")
    print(f"{'='*60}\n")

    # 运行推理
    out_items = state.run(in_pcms)

    # 提取并显示文本
    print(f"\n{'='*60}")
    print(f"推理结果:")
    print(f"{'='*60}\n")

    if out_items:
        text_tokens, _ = out_items[0]
        # 过滤掉 padding token (0) 和特殊 token (3)
        text_tokens_filtered = [t.item() for t in text_tokens if t.item() not in [0, 3]]

        # 解码文本
        decoded_text = text_tokenizer.decode(text_tokens_filtered)

        print(f"转录文本: {decoded_text}")
        print(f"\n文本 tokens ({len(text_tokens_filtered)} 个):")
        print(text_tokens_filtered[:50])  # 显示前 50 个 token
        if len(text_tokens_filtered) > 50:
            print(f"... (还有 {len(text_tokens_filtered) - 50} 个)")
    else:
        print("未生成输出")

    # 读取 ground truth
    import json
    json_file = audio_file.replace(".wav", ".json")
    if Path(json_file).exists():
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            alignments = data.get("alignments", [])
            gt_text = "".join([word for word, _, _ in alignments])
            print(f"\n标准答案: {gt_text}")

if __name__ == "__main__":
    with torch.no_grad():
        main()
