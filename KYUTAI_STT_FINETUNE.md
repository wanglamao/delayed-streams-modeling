# Kyutai STT 中文微调完整指南

**目标**: 在中文带时间戳 ASR 数据上微调 Kyutai STT (Streaming Speech-to-Text) 模型

**最后更新**: 2026-02-06

> 📚 **文档说明**: 本文档整合了原 `MoshiFinetune_微调Kyutai_STT_中文数据_调研与操作指南.md` 和 `KYUTAI_STT1B_中文微调_进展记录.md`，提供从数据准备到推理验证的完整流程。

> 💡 **快速导航**:
> - 新手快速开始: 跳到 [快速开始](#快速开始)
> - 大数据集 (> 100万样本): 查看 [WEBDATASET_GUIDE.md](WEBDATASET_GUIDE.md)
> - Docker 部署: 查看 [DOCKER_FINETUNE_GUIDE.md](DOCKER_FINETUNE_GUIDE.md)
> - 故障排查: 跳到 [故障排除](#故障排除)

---

## 目录

1. [快速开始](#快速开始)
2. [系统架构](#系统架构)
3. [数据格式](#数据格式)
4. [训练配置](#训练配置)
5. [训练流程](#训练流程)
6. [推理验证](#推理验证)
7. [性能优化](#性能优化)
8. [故障排除](#故障排除)
9. [进展记录](#进展记录)

---

## 快速开始

### 前置要求

- Python 3.10+
- PyTorch 2.0+ with CUDA
- 已准备好的中文 ASR 数据（JSONL + alignments）
- GPU: 至少 24GB 显存（推荐）

### 3 步开始训练

```bash
# 1. 准备数据（如果是大数据集 > 100万样本，建议转换为 WebDataset）
conda activate ala
python scripts/convert_to_webdataset.py \
  --input_dir data/stt_zh \
  --output_dir data/stt_zh_webdataset \
  --samples_per_shard 5000

# 2. 修改训练配置
# 编辑 configs/stt_sft_filtered_zh_data.yaml
# 设置 data.train_data 和 run_dir

# 3. 启动训练（2 卡示例）
cd moshi-finetune
torchrun --nproc-per-node 2 -m train ../configs/stt_sft_filtered_zh_data.yaml
```

### 测试 Checkpoint

训练过程中或训练后测试效果：

```bash
conda activate ala

# 单文件测试（详细模式）
python scripts/stt_test_checkpoint.py \
  --checkpoint /path/to/run_dir/checkpoints/checkpoint_000100/consolidated \
  --audio test.wav \
  --verbose

# 批量测试（汇总统计）
python scripts/stt_test_checkpoint.py \
  --checkpoint /path/to/checkpoint/consolidated \
  --audio data/stt_zh/wavs/*.wav \
  --batch
```

详细步骤见下文。

---

## 系统架构

### 1.1 `delayed-streams-modeling` (推理/评估)

- **推理脚本**:
  - `scripts/stt_test_checkpoint.py` - **推荐**: 微调 checkpoint 测试专用，支持 CER 计算和 mode collapse 检测
  - `scripts/stt_from_file_pytorch.py` - 官方推理脚本，适用于原始 HF 模型
  - 加载方式: `moshi.models.loaders.CheckpointInfo.from_hf_repo(...)`

- **配置文件**: `configs/config-stt-en_fr-hf.toml`, `configs/config-stt-en-hf.toml`
  - 包含 `text_tokenizer_file`, `audio_tokenizer_file (mimi)`, `asr_delay_in_tokens`

- **批量评估**: `scripts/stt_evaluate_on_dataset.py`

### 1.2 `moshi-finetune` (训练/LoRA/FSDP)

- **训练入口**: `torchrun ... -m train your.yaml`
- **配置方式**: YAML (示例: `example/moshi_7B.yaml`)
- **数据管线**:
  - JSONL 清单 + 每条 wav 的 `*.json` (含 `alignments`)
  - 读取与切分: `finetune/data/dataset.py`
  - 组装 token 流: `finetune/data/interleaver.py`

**核心思路**: `moshi-finetune` 训练流程本质是对 `moshi.models` 里的 `LMModel` 做 LoRA/全参微调。只需把基座 checkpoint 切到 Kyutai STT (如 `kyutai/stt-1b-en_fr-candle`)，按格式组织数据即可。

---

## 数据格式

### 3.1 训练清单: `train.jsonl`

每行一条音频样本:
```jsonl
{"path": "data/wavs/utt_000001.wav", "duration": 12.34}
{"path": "data/wavs/utt_000002.wav", "duration": 8.91}
```

### 3.2 对齐标注: `utt_000001.json`

训练时对每个 `xxx.wav` 自动读取同名 `xxx.json`:
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

**字段约束** (非常重要):
- `alignments` 必须按 `start` 递增排序 (训练用二分查找)
- 时间单位为秒，相对音频文件起点 (0s)
- `start < end` (否则被过滤)
- `speaker` 建议统一填 `"SPEAKER_MAIN"` (训练默认 `keep_main_only=True`)

### 3.3 音频要求

- **单声道 (mono)**: 必须！立体声会导致 codebook 维度不匹配
- **采样率**: 建议预处理到 24kHz (Mimi 采样率)
- **格式**: WAV (PCM 16-bit 或 float32)

### 3.4 数据自检

训练前建议运行自检脚本:
```bash
python scripts/validate_stt_dataset.py data/stt_zh/train.jsonl
```

### 3.5 大数据集: 使用 WebDataset

**适用场景**: 训练数据 > 100 万条音频

详见 [WEBDATASET_GUIDE.md](WEBDATASET_GUIDE.md)

简要步骤:
```bash
# 转换
python scripts/convert_to_webdataset.py \
  --input_dir data/stt_zh \
  --output_dir data/stt_zh_webdataset \
  --samples_per_shard 5000

# 配置中启用
data:
  train_data: "data/stt_zh_webdataset"
  use_webdataset: true
```

---

## 训练配置

### 4.1 选择基座模型

**推荐**: `kyutai/stt-1b-en_fr-candle` (1B 参数，低延迟)

```yaml
moshi_paths:
  hf_repo_id: "kyutai/stt-1b-en_fr-candle"
```

**备选**: `kyutai/stt-2.6b-en-candle` (2.6B 参数，更高精度，但需更多显存)

### 4.2 最小训练配置

创建 `stt_zh_lora.yaml`:

```yaml
# 数据配置
data:
  train_data: "/path/to/data/stt_zh_webdataset"  # 或 train.jsonl
  use_webdataset: true                           # 大数据集建议启用
  eval_data: ""
  shuffle: true

# 模型配置
moshi_paths:
  hf_repo_id: "kyutai/stt-1b-en_fr-candle"

# 输出目录
run_dir: "/path/to/runs/stt_zh_lora_run1"

# LoRA 配置 (推荐，避免全参数微调不稳定)
full_finetuning: false
lora:
  enable: true
  rank: 64
  scaling: 2.0
  ft_embed: false

# 训练超参
duration_sec: 20              # 每个训练片段时长
batch_size: 2                 # 根据显存调整
max_steps: 2000

gradient_checkpointing: true
max_norm: 1.0                 # 梯度裁剪，防止梯度爆炸

optim:
  lr: 2.0e-5
  weight_decay: 0.1
  pct_start: 0.05

# 日志和检查点
log_freq: 1
do_eval: false
do_ckpt: true
ckpt_freq: 100

# 推荐合并保存权重 (方便推理)
save_adapters: false          # LoRA 模式下可设为 true 只保存 adapter

# ⚠️ STT 特有配置 (非常重要！)
interleaver:
  downmix_to_mono: true       # 自动转单声道
  audio_delay_sec: 0.5        # ⚠️ 必须匹配原始模型的 audio_delay_seconds (见 config.json)

text_padding_weight: 0.1      # ⚠️ Padding token loss 权重 (不要过高，否则 mode collapse)
```

**⚠️ 关键参数说明**:

- `audio_delay_sec`: **必须与原始模型的 `audio_delay_seconds` 一致**！
  - `kyutai/stt-1b-en_fr-candle`: 0.5s (查看 checkpoint 的 `config.json`)
  - `kyutai/stt-2.6b-en-candle`: 2.5s
  - 错配会导致音文对齐失败和 mode collapse (见 [8.2 故障排除](#82-严重问题-模型-mode-collapse-模态崩溃))

- `text_padding_weight`: 控制 padding token 的 loss 权重
  - 推荐值: 0.1 (不要超过 0.5)
  - 过高会导致模型依赖 padding token 而非学习真实语音

- `full_finetuning: false` + `lora.enable: true`: **强烈推荐**
  - 全参数微调 (1B+ 模型) 容易不稳定
  - LoRA 更稳定且节省显存

### 4.3 显存优化配置

**24GB 显存** (单卡):
```yaml
duration_sec: 15
batch_size: 2
gradient_checkpointing: true
lora:
  rank: 32
```

**48GB+ 显存** (单卡):
```yaml
duration_sec: 20
batch_size: 4
gradient_checkpointing: true
lora:
  rank: 64
```

---

## 训练流程

### 5.1 环境准备

```bash
# 激活环境
conda activate ala

# 安装依赖 (首次)
cd moshi-finetune
pip install -e .
```

### 5.2 下载基座模型 (可选)

如果需要本地模型文件:
```bash
python scripts/hf_download.py kyutai/stt-1b-en_fr-candle
```

模型会下载到 `hf_models/kyutai/stt-1b-en_fr-candle/`

### 5.3 启动训练

**单卡训练**:
```bash
cd moshi-finetune
torchrun --nproc-per-node 1 -m train example/stt_zh_lora.yaml
```

**多卡训练** (2 卡):
```bash
torchrun --nproc-per-node 2 -m train example/stt_zh_lora.yaml
```

**Docker 训练** (推荐生产环境):
```bash
docker compose up sft
```

详见 [DOCKER_FINETUNE_GUIDE.md](DOCKER_FINETUNE_GUIDE.md)

### 5.4 监控训练

训练日志保存在 `{run_dir}/logs/`

关键指标:
- `text_loss`: 文本预测 loss
- `audio_loss`: 音频 token loss (如果启用)
- `learning_rate`: 学习率变化

---

## 推理验证

### 6.1 快速测试推理 (推荐)

使用封装好的测试脚本 `scripts/stt_test_checkpoint.py`:

```bash
conda activate ala

# 测试单个音频文件
python scripts/stt_test_checkpoint.py \
  --checkpoint /path/to/run_dir/checkpoints/checkpoint_000100/consolidated \
  --audio test.wav

# 详细模式（显示 token 分析，检测 mode collapse）
python scripts/stt_test_checkpoint.py \
  --checkpoint /path/to/checkpoint/consolidated \
  --audio test.wav \
  --verbose

# 批量测试（显示汇总统计和平均 CER）
python scripts/stt_test_checkpoint.py \
  --checkpoint /path/to/checkpoint/consolidated \
  --audio data/stt_zh/wavs/*.wav \
  --batch
```

**脚本特性**:
- ✓ 自动查找 mimi 和 tokenizer 权重（无需手动指定）
- ✓ 自动加载标准答案（同名 .json 文件）并计算 CER
- ✓ Token 多样性检测（自动警告 mode collapse）
- ✓ 支持批量测试和汇总统计（平均 CER、整体 token 分布）
- ✓ 支持通配符（如 `*.wav`）批量处理

**参数说明**:
- `--checkpoint`: checkpoint 的 consolidated 目录路径
- `--audio`: 音频文件路径（支持多个文件和通配符）
- `--verbose`: 显示详细信息（token IDs、多样性分析等）
- `--batch`: 批量模式（简洁输出，显示汇总统计）
- `--mimi-weight`: (可选) 手动指定 Mimi 权重路径
- `--tokenizer`: (可选) 手动指定 tokenizer 路径
- `--device`: 运行设备（默认 cuda）

**输出示例**:

```
[转录结果]
  加快具有中国特色的原始性创新技术

[标准答案]
  加快具有中国特色的原始性创新技术与装备的研发

[字符错误率 (CER)]
  8.33%
  ⚠️  有少量错误

[Token 分析]
  唯一 Tokens: 235
  ✓ Token 多样性正常
```

### 6.2 使用官方脚本推理

Checkpoint 路径:
```
{run_dir}/checkpoints/checkpoint_000100/consolidated/consolidated.safetensors
```

推理命令:
```bash
conda activate ala
python scripts/stt_from_file_pytorch.py \
  --hf-repo kyutai/stt-1b-en_fr-candle \
  --moshi-weight runs/stt_zh_lora_run1/checkpoints/checkpoint_000100/consolidated/consolidated.safetensors \
  test_audio.wav
```

### 6.3 LoRA Adapter 推理

如果 `save_adapters: true`，需要修改推理脚本支持 LoRA 加载 (当前未实现)。

**建议**: 训练时使用 `save_adapters: false`，直接保存合并权重。

### 6.4 批量评估

```bash
conda activate ala
python scripts/stt_evaluate_on_dataset.py \
  --dataset your_test_set \
  --hf-repo kyutai/stt-1b-en_fr-candle \
  --moshi-weight runs/stt_zh_lora_run1/checkpoints/checkpoint_final/consolidated/consolidated.safetensors
```

注意: 当前评估脚本偏英文，中文评估需要:
- 使用 CER (字错误率) 而非 WER
- 自定义中文文本规范化 (标点、全角半角等)

---

## 性能优化

### 7.1 WebDataset (海量数据优化)

**适用场景**: > 100 万条音频

详见 [WEBDATASET_GUIDE.md](WEBDATASET_GUIDE.md)

**优势**:
- 避免文件系统瓶颈 (inode 耗尽)
- 减少网络往返 (NFS/HDFS)
- 分布式训练友好
- 存储压缩 ~98%

### 7.2 训练加速技巧

1. **使用 WebDataset** (大数据集)
2. **梯度检查点**: `gradient_checkpointing: true`
3. **混合精度**: 默认已启用 bfloat16
4. **预处理音频**: 统一采样率和声道数
5. **合理 shard 大小**: WebDataset `samples_per_shard=5000`

---

## 故障排除

### 8.1 常见错误

**问题**: `Shape mismatch: expected 32 codebooks, got 64`

**原因**: 音频是立体声，被当成 2 路 batch 维

**解决**:
```yaml
interleaver:
  downmix_to_mono: true
```

---

**问题**: `CUDA out of memory`

**解决**:
1. 减小 `batch_size`
2. 减小 `duration_sec`
3. 启用 `gradient_checkpointing: true`
4. 降低 LoRA `rank`

---

**问题**: 时间戳系统性偏移

**原因**: DSM 固定延迟 (`asr_delay_in_tokens`)

**解决**:
```yaml
interleaver:
  audio_delay_sec: 0.48  # 根据模型调整 (约 asr_delay_in_tokens / frame_rate)
```

---

**问题**: WebDataset 找不到 tar 文件

**检查**:
```bash
ls data/stt_zh_webdataset/*.tar
python scripts/inspect_webdataset.py --data_dir data/stt_zh_webdataset
```

---

### 8.2 严重问题: 模型 Mode Collapse (模态崩溃)

**症状**: 微调后模型只生成重复的无意义 token，完全无法识别语音

**发现过程** (2026-02-06):

测试了多个 checkpoint (run2/run3 的 step 500/1000)，所有模型均出现严重 mode collapse:

```bash
# 测试音频: emilia_zh_0000983244.wav
# 真实标注: "加快具有中国特色的原始性创新技术与装备的研发实现传统食品传统餐饮的工业化标准化"

# 模型输出 (所有 checkpoint):
Token IDs: [260, 260, 233, 260, 233, 260, 233, ...]  # 只有 260 (空格) 和 233 (<0xE5>)
解码结果: "  �  �  �  �  �  �  �  �  �  �  �  �"  # 完全无效
```

**对比测试**: 原始英法 STT 模型在相同音频上能输出部分中文词汇 (虽不完美但可识别)，证明 tokenizer 支持中文且音频无问题。

**推理测试脚本**:

```bash
conda activate ala

# 推荐：使用封装脚本（自动查找权重，显示 token 分析）
python scripts/stt_test_checkpoint.py \
  --checkpoint /path/to/checkpoint/consolidated \
  --audio test_audio.wav \
  --verbose

# 或使用官方脚本
python scripts/stt_from_file_pytorch.py \
  --hf-repo kyutai/stt-1b-en_fr-candle \
  --moshi-weight /path/to/checkpoint/consolidated/consolidated.safetensors \
  test_audio.wav
```

**根因分析**:

训练配置检查发现多个可能导致崩溃的问题:

| 问题 | run2 配置 | run3 配置 | 影响 |
|------|----------|----------|------|
| **audio_delay_sec** | 0.8 秒 | 1.0 秒 | 原始模型 config.json 中 `audio_delay_seconds: 0.5`，训练时错位导致音文对齐失败 |
| **text_padding_weight** | 0.5 (隐式) | 0.5 (隐式) | Padding token loss 权重过高，模型学会依赖 padding |
| **full_finetuning** | true | true | 全参数微调不稳定，容易破坏预训练权重 |
| **learning_rate** | 0.0002 | 2e-5 | run2 初始 lr 过高导致 step 1000 后 loss=NaN |
| **batch_size** | 128 | 64 | run2 batch 过大可能加剧不稳定性 |

**训练指标检查**:

```bash
# run2: Loss 快速下降但出现 NaN
step 100:  loss=3.16, text_loss=3.16
step 500:  loss=1.15, text_loss=1.15
step 1000: loss=0.80, text_loss=0.80
step 1060: loss=NaN, text_loss=NaN  # 崩溃

# run3: Loss 正常下降，无 NaN，但仍 mode collapse
step 100:  loss=3.02, text_loss=3.02, perplexity=2.23
step 500:  loss=0.88, text_loss=0.88, perplexity=1.84
step 1000: loss=0.84, text_loss=0.84, perplexity=1.79
```

**解决方案** (已应用到 `configs/stt_sft_filtered_zh_data.yaml`):

1. **修正 audio_delay_sec**:
   ```yaml
   interleaver:
     audio_delay_sec: 0.5  # 匹配原始模型的 audio_delay_seconds
   ```

2. **降低 text_padding_weight**:
   ```yaml
   text_padding_weight: 0.1  # 从 0.5 降低到 0.1，避免过度依赖 padding token
   ```

3. **切换到 LoRA 训练**:
   ```yaml
   full_finetuning: false
   lora:
     enable: true
     rank: 64
     scaling: 2.0
     ft_embed: false
   save_adapters: true  # LoRA 模式下保存 adapter
   ```

4. **添加梯度裁剪**:
   ```yaml
   max_norm: 1.0  # 防止梯度爆炸
   ```

5. **更新 run_dir**:
   ```yaml
   run_dir: "/nvmedata/ala_storage/ala_runs/stt_zh_filtered_2_5m_run4_lora"
   ```

**验证步骤**:

重新训练后，在 step 100、500、1000 分别测试:

```bash
conda activate ala

# 使用封装脚本测试
python scripts/stt_test_checkpoint.py \
  --checkpoint /nvmedata/ala_storage/ala_runs/stt_zh_filtered_2_5m_run4_lora/checkpoints/checkpoint_000100/consolidated \
  --audio delayed-streams-modeling/data/stt_zh/wavs/emilia_zh_0000983244.wav \
  --verbose

# 预期结果:
# - Token 多样性显著提高 (不再只有 260, 233)
# - 能输出部分中文汉字 (即使有错误)
# - Loss 稳定下降，无 NaN
```

**关键教训**:

1. **音文对齐参数至关重要**: `audio_delay_sec` 必须匹配原始模型的 `audio_delay_seconds`
2. **Padding token 权重不能过高**: 会导致模型学会依赖 padding 而非真正识别
3. **全参数微调风险大**: 对于大模型 (1B+)，LoRA 更稳定
4. **Training loss 不能作为唯一指标**: run3 loss 正常但模型完全无效，必须做推理验证
5. **Baseline 对比必不可少**: 原始模型测试确认了 tokenizer 和数据无问题

---

### 8.3 调试技巧

1. **先跑小数据集** (100-1000 样本)
2. **检查数据格式**: `validate_stt_dataset.py`
3. **验证 WebDataset**: `inspect_webdataset.py --verify`
4. **查看详细日志**: `log_freq: 1`
5. **单步调试**: 在训练代码中加 `import pdb; pdb.set_trace()`
6. **定期推理验证**: 不要只看 loss，在 checkpoint 上实际测试音频输出
7. **对比原始模型**: 用原始权重测试相同音频，确认 baseline 行为

---

## 进展记录

### 已完成 ✓

日期: 2026-02-06

- [x] 数据格式调研和对齐方案
- [x] 本地 HF 模型下载脚本 (`scripts/hf_download.py`)
- [x] 数据自检脚本 (`scripts/validate_stt_dataset.py`)
- [x] 训练配置模板 (`moshi-finetune/example/stt_zh_lora*.yaml`)
- [x] STT 特有选项暴露 (`interleaver.downmix_to_mono`, `audio_delay_sec`)
- [x] **WebDataset 完整实现**:
  - 数据转换脚本 (`scripts/convert_to_webdataset.py`)
  - 检查验证脚本 (`scripts/inspect_webdataset.py`)
  - 训练数据加载器 (`moshi-finetune/finetune/data/webdataset_loader.py`)
  - 微调测试验证通过
  - 完整使用文档 (`WEBDATASET_GUIDE.md`)
- [x] Docker 训练环境 (`Dockerfile`, `docker-compose.yml`)
- [x] 完整文档整合

### 当前状态

**训练实验**:

| Run ID | 数据规模 | 配置 | 状态 | 结果 |
|--------|---------|------|------|------|
| run2 | 2.5M 样本 | Full finetune, lr=0.0002, batch=128 | ❌ 失败 | Step 1000 后 loss=NaN，mode collapse |
| run3 | 2.5M 样本 | Full finetune, lr=2e-5, batch=64 | ❌ 失败 | Loss 正常但 mode collapse (audio_delay_sec=1.0 错配) |
| run4_lora | 2.5M 样本 | LoRA, audio_delay_sec=0.5, text_padding_weight=0.1 | 🔄 待运行 | 预期: 音文对齐正确，训练稳定 |

**问题诊断完成** ✓:
- 识别 mode collapse 根因 (audio_delay 错配 + padding weight 过高 + 全参数不稳定)
- 完成配置修正 (`configs/stt_sft_filtered_zh_data.yaml`)
- 文档化故障排查流程 (见 [8.2 模型 Mode Collapse](#82-严重问题-模型-mode-collapse-模态崩溃))

**待实际训练验证**:
- [ ] 在 run4_lora 配置上训练并验证收敛性
- [ ] 确认 LoRA 训练消除 mode collapse
- [ ] 时间戳对齐精度验证
- [ ] 中文 CER 评测

### 未来计划

1. **音频预编码方案** (可选优化)
   - 预先对 WAV 做 Mimi 编码，生成二进制 records
   - 训练时跳过 Mimi 编码，加速 30-50%
   - 适用于超大数据集 (> 1 亿样本) 且需要多轮训练的场景
   - 注: WebDataset 已解决大部分 I/O 瓶颈，此优化优先级较低

2. **中文评测工具链**
   - 实现 CER 计算
   - 中文文本规范化
   - 时间戳对齐评估

3. **多语言支持**
   - 中英混合数据训练
   - 代码切换 (code-switching) 支持

---

## 关键决策清单

训练前需确认:

- [ ] **时间戳粒度**: 字级 / 词级 / 句子片段？
- [ ] **音频类型**: 短 utterance (单句) / 长录音 (多句连续)？
- [ ] **音频规格**: 采样率、声道数 (建议: 24kHz mono)
- [ ] **期望输出**: 字级 / 词级 / 句级时间戳？
- [ ] **数据规模**: < 100万 (JSONL) / > 100万 (WebDataset)
- [ ] **分词策略**: 中文是否需要预分词？

> 备注: DSM 固定延迟 (`asr_delay_in_tokens`) 可能导致时间戳偏移，后续需记录偏移量和补偿方案。

---

## 参考资料

- [Moshi 主仓库](https://github.com/kyutai-labs/moshi)
- [Kyutai STT Issue #4](https://github.com/kyutai-labs/delayed-streams-modeling/issues/4) - STT 微调讨论
- [WebDataset 使用指南](WEBDATASET_GUIDE.md)
- [Docker 训练指南](DOCKER_FINETUNE_GUIDE.md)
- [WebDataset 官方文档](https://github.com/webdataset/webdataset)
