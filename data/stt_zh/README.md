# 中文 STT 微调数据目录（示例结构）

本目录用于存放“带时间戳 ASR 对齐”的中文训练数据（供 `moshi-finetune` 使用）。

推荐结构：
```
data/stt_zh/
  train.jsonl
  wavs/
    utt_000001.wav
    utt_000001.json
    utt_000002.wav
    utt_000002.json
    ...
```

## 1) `train.jsonl`
每行一条音频样本（至少包含 `path`、`duration`）：
```jsonl
{"path": "data/stt_zh/wavs/utt_000001.wav", "duration": 12.34}
{"path": "data/stt_zh/wavs/utt_000002.wav", "duration": 8.91}
```

建议 `path` 按“仓库根目录”写相对路径（便于从不同工作目录启动训练/校验脚本）。

## 2) `utt_000001.json`
每条 wav 同名 sidecar JSON，至少包含 `alignments`：
```json
{
  "alignments": [
    ["你", [0.12, 0.18], "SPEAKER_MAIN"],
    ["好", [0.18, 0.24], "SPEAKER_MAIN"]
  ]
}
```

约束（关键）：
- `alignments` 必须按 `start` 递增排序
- 时间单位：秒；相对该 wav 起点（0s）
- `start < end`
- `speaker` 建议统一 `"SPEAKER_MAIN"`（默认 `keep_main_only: true`）

## 3) 音频建议
- 尽量 mono（单声道）。若源数据为 stereo，训练侧会 downmix（见 `moshi-finetune/example/stt_zh_lora.yaml` 的 `interleaver.downmix_to_mono`）。
- 采样率建议统一到 Mimi 的采样率（通常 24kHz）以减少训练时重采样开销。

## 4) Smoke 样例（用于自检）
本目录下默认生成了一个极小样例用于跑通 `scripts/validate_stt_dataset.py`：
- `data/stt_zh/train.jsonl`
- `data/stt_zh/wavs/smoke_000001.wav`
- `data/stt_zh/wavs/smoke_000001.json`

实际训练前请替换为你的真实中文数据清单与标注。
