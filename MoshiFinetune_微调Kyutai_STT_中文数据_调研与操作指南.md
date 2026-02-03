# 用 `moshi-finetune` 微调 Kyutai STT（中文数据）调研与操作指南

目标：在中文带时间戳 ASR 数据上微调 Kyutai STT（Streaming Speech-to-Text）模型，并能用本仓库的推理脚本验证效果。

相关仓库路径（你本机）：
- 推理/评估/配置：`/mnt/d/nashome/macworkspace/delayed-streams-modeling`
- 训练框架：`/mnt/d/nashome/macworkspace/moshi-finetune`

---

## 1. 两边代码分别负责什么

### 1.1 `delayed-streams-modeling`（推理/评估）
- 推理脚本：`scripts/stt_from_file_pytorch.py`
  - 通过 `moshi.models.loaders.CheckpointInfo.from_hf_repo(...)` 加载 STT 模型与 tokenizer/mimi。
- 服务/配置：`configs/config-stt-en_fr-hf.toml`、`configs/config-stt-en-hf.toml`
  - 包含 `text_tokenizer_file`、`audio_tokenizer_file(mimi)`、`asr_delay_in_tokens` 等关键参数。
- 批量评估（目前偏英文）：`scripts/stt_evaluate_on_dataset.py`

### 1.2 `moshi-finetune`（训练/LoRA/FSDP）
- 训练入口：`torchrun ... -m train your.yaml`（实现文件：`moshi-finetune/train.py`）
- 配置方式：YAML（示例：`moshi-finetune/example/moshi_7B.yaml`）
- 数据管线：
  - JSONL 清单 + 每条 wav 的 `*.json`（含 `alignments`）
  - 读取与切分：`moshi-finetune/finetune/data/dataset.py`
  - 组装 token 流：`moshi-finetune/finetune/data/interleaver.py`

结论：`moshi-finetune` 的训练主流程本质上是对 `moshi.models` 里的 `LMModel` 做 LoRA/全参微调；只要把基座 checkpoint 切到 Kyutai STT（例如 `kyutai/stt-1b-en_fr-candle`），并把数据按它的输入组织好，就有机会直接跑起来（但仍有若干“可能需要对齐/改动”的点，见最后一节）。

---

## 2. 数据格式：把你的“带时间戳 ASR 数据”对齐到 `moshi-finetune` 期望的格式

### 2.1 训练清单：`train.jsonl`
每行一条音频样本，至少包含：
```jsonl
{"path": "data/wavs/utt_000001.wav", "duration": 12.34}
{"path": "data/wavs/utt_000002.wav", "duration": 8.91}
```

### 2.2 每条音频旁边的标注：`utt_000001.json`
训练时会对每个 `xxx.wav` 自动读取同名 `xxx.json`，至少需要：
```json
{
  "alignments": [
    ["你", [0.12, 0.18], "SPEAKER_MAIN"],
    ["好", [0.18, 0.24], "SPEAKER_MAIN"],
    ["，", [0.24, 0.30], "SPEAKER_MAIN"],
    ["今", [0.55, 0.62], "SPEAKER_MAIN"],
    ["天", [0.62, 0.70], "SPEAKER_MAIN"]
  ]
}
```

字段约束（非常重要）：
- `alignments` 必须按 `start` 递增排序（训练里用二分定位切片）。
- 时间单位为秒，且相对当前音频文件起点（0s）。
- `start < end`（不然会被过滤掉）。
- `speaker` 建议统一填 `"SPEAKER_MAIN"`（训练默认 `keep_main_only=True`，只保留主说话人）。

### 2.3 音频要求（建议）
- **单声道（mono）**：对 STT 微调最稳。
  - `moshi-finetune` 的 tokenizer 会把“音频通道数”当成 batch 维来编码；若你给了立体声，它可能被当成 2 路并展平，从而导致 codebook 维度不匹配 STT（STT 期望 32 codebooks）。
- 采样率：建议预处理到 Mimi 的采样率（通常 24kHz），避免训练时每步重采样的额外开销。

### 2.4 Moshi 是如何“使用带时间戳 ASR 数据”的（代码级机制）

一句话：Moshi **不会**把时间戳当成连续值去回归；它把你的 `alignments`（文本 + 起止时间）转成一个与音频 tokenizer 时间步对齐的 **离散文本 token 流**，再把这个文本流与 Mimi 的音频 token 流拼成多流序列喂给模型训练。

核心发生在 `moshi-finetune/finetune/data/interleaver.py`：

#### Step A：按 `duration_sec` 把一条 wav 切成多个训练片段
- 数据迭代器用 `sphn.dataset_jsonl(..., duration_sec=instruct_tokenizer.duration_sec, ...)` 把长音频按固定时长切片（见 `moshi-finetune/finetune/data/dataset.py`）。
- 每个片段会带上 `start_time_sec`（片段起点，相对整条音频的秒数）和 `path`。

#### Step B：从整条音频的 `alignments` 中裁剪出“当前片段范围”的对齐文本
`InterleavedTokenizer.__call__(wav, start_sec, path)` 会：
1. 读取同名 `path.with_suffix(".json")` 的 `alignments`。
2. 用二分函数 `dicho(...)` 找到 `[start_sec, start_sec + duration_sec]` 内的对齐条目。
3. 把时间戳平移到片段坐标系：`(start, end) -> (start - start_sec, end - start_sec)`（见 `moshi-finetune/finetune/data/interleaver.py`）。

#### Step C：把时间戳对齐文本变成“与音频时间步同长度”的文本 token 序列
`Interleaver.prepare_item(alignments, segment_duration)` 会把 `alignments` 变成一个长度为 `T` 的 1D token 流（shape 最终是 `[1, 1, T]`），其中每个位置对应一个“音频 tokenizer 的时间步”。

关键点：
- 文本分词使用 SentencePiece：每个 `alignment[i][0]`（文本片段）会被编码成若干子词 id（见 `tokenize(...)` 与 `_tokenize(...)`）。
- “时间 -> 帧”的映射方式：
  - 设 `audio_frame_rate` 为音频 tokenizer 的帧率（训练中从 `mimi.frame_rate` 传入）。
  - 文本 token 序列长度 `T = ceil(segment_duration * audio_frame_rate)`（`build_token_stream(...)`）。
  - 当某个词的开始时间 `start_sec` 满足 `start_sec * audio_frame_rate < t + 1` 时，表示它应该在第 `t` 个时间步附近开始“吐 token”（简化理解：把秒数乘帧率变成帧索引）。
- 训练用到的几个特殊 token（由模型 config 提供 id）：
  - `text_padding`: 没有文本时的填充
  - `end_of_text_padding`: **词边界**标记（推理时用它定位“词/片段”的时间戳）
  - `in_word_padding`: 词持续时间内的填充（可默认等同 padding，也可单独区分）
  - `zero_padding`: 用于“不要 embedding/静音”的占位
- 词边界是怎么形成的：
  - 当要开始输出一个词的 token 时，会把前一帧从 padding/in_word_padding 改成 `end_of_text_padding`，作为边界标记（见 `build_token_stream(...)` 中对 `t-1` 的赋值）。

> 直观理解：你的时间戳监督的是“这个词大概从哪一帧开始”，而不是监督一个连续时间回归头。

#### Step D：把“文本流”与 Mimi “音频流”拼成多流序列喂给模型
在同一个 `InterleavedTokenizer.__call__` 里：
1. Mimi 把 wav 编成离散音频 tokens（多个 codebooks × T 帧）。
2. 文本 token 流被 pad/truncate 到同样的 `T`。
3. 沿 codebook 维拼接：`codes = cat([text_tokens, audio_tokens], dim=1)`。
   - 形状大致是 `[B, 1 + audio_codebooks, T]`（文本 1 路 + 音频多路）。

#### Step E：训练时如何用这些 token（loss）
训练循环（`moshi-finetune/train.py`）会对模型输出分别算：
- `text_loss`: 用 `output.text_logits` 去预测 `codes[:, :model.audio_offset]`（文本流的 token），并对 padding 类 token 降权（`text_padding_weight`）。
- `audio_loss`: 用 `output.logits` 去预测音频 codebooks（并可对第一个 codebook 加权 `first_codebook_weight_multiplier`）。

对“ASR 微调”的含义：
- 你提供的时间戳文本对齐，决定了文本 token 在时间轴上出现的位置；
- 模型通过交错序列学习“从音频 token（以及历史上下文）预测后续文本 token”，同时可能也继续学习音频自回归部分（取决于模型/配置）。

---

## 3. 训练：用 `moshi-finetune` 跑 LoRA（先跑通再追效果）

### 3.1 关键想法：把基座从 Moshi 换成 Kyutai STT
在 `moshi-finetune` 配置里把：
```yaml
moshi_paths:
  hf_repo_id: "kyutai/moshiko-pytorch-bf16"
```
改成：
```yaml
moshi_paths:
  hf_repo_id: "kyutai/stt-1b-en_fr-candle"
```

说明：
- `stt-1b-en_fr-candle` 是 1B 流式 STT，延迟小，通常更适合先验证流程。
- 2.6B 也可以（`kyutai/stt-2.6b-en-candle`），但更吃显存与吞吐。

### 3.2 建议先用一个最小可跑的 YAML
从 `moshi-finetune/example/moshi_7B.yaml` 复制一份，比如 `stt_zh_lora.yaml`，然后至少改这些：
```yaml
data:
  train_data: "/abs/path/to/train.jsonl"
  eval_data: ""
  shuffle: true

moshi_paths:
  hf_repo_id: "kyutai/stt-1b-en_fr-candle"

run_dir: "/abs/path/to/runs/stt_zh_lora_run1"

full_finetuning: false
lora:
  enable: true
  rank: 64
  scaling: 2.
  ft_embed: false

# 先把 seq 长度 / batch 调小跑通
duration_sec: 20
batch_size: 2
max_steps: 2000

gradient_checkpointing: true
optim:
  lr: 2e-6
  weight_decay: 0.1
  pct_start: 0.05

log_freq: 1
do_eval: false
do_ckpt: true
ckpt_freq: 100

# 为了后续推理最省事，建议先合并保存（而不是只存 LoRA）
save_adapters: false
```

### 3.3 启动训练
在 `moshi-finetune` 目录：
```bash
torchrun --nproc-per-node 1 -m train stt_zh_lora.yaml
```

如果你用 `uv` 管环境（官方推荐的用法），则：
```bash
uv run torchrun --nproc-per-node 1 -m train stt_zh_lora.yaml
```

### 3.4 微调 Kyutai STT 需要修改代码吗？

通常 **不需要**（先跑通的最短路径）：
- 只改 YAML：把 `moshi_paths.hf_repo_id` 指向 `kyutai/stt-1b-en_fr-candle`（或 2.6B 版本），并填好 `data.train_data`、`run_dir`。
- 把训练音频预处理成 **mono**，标注按 `jsonl + alignments.json` 格式准备好。
- 训练时先用 `save_adapters: false`（输出合并权重），这样推理验证不需要额外改脚本。

但下面这些情况**大概率需要改代码**（或做明确的额外处理）：
- **你的音频是 stereo/多通道**：当前训练 tokenizer 会把“通道维”当 batch 维编码并展平，容易导致与 STT 的 32 codebooks 不匹配；建议预处理 downmix 到 mono，或在 `InterleavedTokenizer` 里强制 downmix。
- **你想只保存 LoRA（`save_adapters: true`）并用本仓库脚本推理**：`delayed-streams-modeling/scripts/stt_from_file_pytorch.py` 目前没有 `--lora-weight` 的加载逻辑，需要补。
- **你对“时间戳严格对齐”有要求**：STT 有 DSM 固定延迟（`asr_delay_in_tokens`）；如果训练后整体偏移，需要在数据侧做 offset，或在训练侧把 `Interleaver(audio_delay=...)` 暴露成可配置参数。
- **训练时报 shape / key mismatch**：例如输出 logits/mask 与训练脚本预期不一致，这时才需要针对报错去改 loss 或模型包装（先跑最小样本最容易定位）。

---

## 4. 训练后验证：用本仓库脚本跑一条音频看看

### 4.1 最推荐的验证方式（合并权重 `save_adapters: false`）
`moshi-finetune` 的 checkpoint 会在类似目录下：
```
{run_dir}/checkpoints/checkpoint_000100/consolidated/consolidated.safetensors
{run_dir}/checkpoints/checkpoint_000100/consolidated/config.json
```

然后在 `delayed-streams-modeling` 目录用推理脚本（复用 HuggingFace 上的 tokenizer/mimi，只覆盖 moshi 权重）：
```bash
uv run scripts/stt_from_file_pytorch.py \
  --hf-repo kyutai/stt-1b-en_fr-candle \
  --moshi-weight /abs/path/to/consolidated.safetensors \
  /abs/path/to/test.wav
```

> 如果不使用 `uv`，确保你当前 python 环境里有 `moshi` 包以及脚本依赖（见 `scripts/stt_from_file_pytorch.py` 文件头的依赖注释）。

### 4.2 如果你只保存 LoRA（`save_adapters: true`）
当前 `delayed-streams-modeling/scripts/stt_from_file_pytorch.py` **没有**直接加载 LoRA adapter 的参数；
你有两条路：
1) 训练时先用 `save_adapters: false`（最省事）；
2) 给推理脚本补上 `--lora-weight` 的加载逻辑（需要改一点代码）。

---

## 5. 可能需要改动/对齐的点（跑通后再处理）

### 5.1 DSM 延迟与“时间戳偏移”问题（很常见）
STT 模型有固定延迟（配置里 `asr_delay_in_tokens`，例如 `configs/config-stt-en_fr-hf.toml` 里是 6）。如果你发现：
- 文本整体偏早 / 偏晚
- 或者词边界时间戳系统性漂移

通常需要对齐其中一个环节：
- **数据侧**：把 `alignments` 的时间整体加/减一个 offset（候选 offset：`asr_delay_in_tokens / mimi.frame_rate`，1B 大约 0.48s）。
- **训练侧**：`moshi-finetune` 的 `Interleaver` 其实支持 `audio_delay` 参数，但当前训练代码未暴露到 YAML（可以在 `moshi-finetune/train.py` 构造 `Interleaver(...)` 时传入）。

建议做法：先不动 offset，把模型跑起来；再用少量样本检查“输出 token 何时开始出现”和“时间戳偏移量”，最后再决定改哪一边。

### 5.2 音频通道数（mono vs stereo）
如上所述，`moshi-finetune/finetune/data/interleaver.py` 会把输入的 `wav` 直接送进 Mimi，并把输出展平做 codebooks：
- 对 STT：请尽量保证输入 wav 是 **单声道**。
- 若你的数据源不可控（经常是双声道录音），可以：
  - 数据预处理阶段统一 downmix 到 mono；
  - 或改 `InterleavedTokenizer.__call__`，检测多通道时做平均再 encode。

### 5.3 中文 tokenizer 与“词级时间戳”的定义
`SentencePiece` 采用子词；中文通常没有空格：
- 能否训练：一般没问题（尤其 tokenizer 里有 byte fallback 时）。
- “词级”时间戳：可能会退化成“子词/字片段”时间戳，需要你明确产品/评测定义：
  - 你希望输出“字级时间戳”、还是“分词后的词级时间戳”？
  - 若要词级：建议在数据准备阶段对中文做分词并在 `alignments` 里按“词”写入（同时维护时间戳）。

### 5.4 中文评测（CER/WER 与规范化）
本仓库的 `scripts/stt_evaluate_on_dataset.py` 更偏英文数据集与英文规范化；
做中文时通常要：
- 用 CER（或自定义分词 + WER）
- 做中文标点/全角半角/数字等规范化

---

## 6. 你需要确认的 4 个信息（决定我后续怎么帮你“把转换脚本/offset 定死”）
1) 你的时间戳粒度：字 / 词 / 句子片段？
2) 你的音频样本类型：每条一个 utterance（短句）还是长录音（多句连续）？
3) 音频当前的采样率与声道数（是否普遍是 stereo）？
4) 你最终想要的输出时间戳：字级、词级，还是仅句级？
