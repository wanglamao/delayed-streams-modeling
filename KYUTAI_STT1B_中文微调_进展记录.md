# Kyutai STT 1B（`kyutai/stt-1b-en_fr-candle`）中文数据微调：进展记录

日期：2026-02-03  
仓库：`/mnt/d/nashome/macworkspace/delayed-streams-modeling`

## 0. 目标与约束

### 目标
- 使用本目录下的 `moshi-finetune/` 对 Kyutai Streaming STT 1B（DSM）进行中文数据微调（优先 LoRA，先跑通再追效果）。
- 微调后使用本仓库 `scripts/stt_from_file_pytorch.py` 对单条音频做推理验证（必要时再扩展批量评测）。

### 强约束（按你的要求）
- HuggingFace 模型文件必须落盘到当前目录（项目内），不使用 `~/.cache` 等全局缓存目录。
- 目标、步骤、关键决策与进展必须持续记录在本文档里。

## 1. 参考资料（必须）
- 操作指南：`MoshiFinetune_微调Kyutai_STT_中文数据_调研与操作指南.md`
- 讨论：`kyutai-labs/delayed-streams-modeling` Issue #4（Fine-tuning Kyutai Speech-To-Text）
  - 该 issue 主题聚焦 STT 微调可行性；正文强调此 issue 仅讨论 STT（TTS 另见 #64）。
  - 结论：本次采用 `moshi-finetune` 跑通 STT 微调流程，并针对 DSM 延迟/数据格式做对齐与记录。

## 2. 当前进展

### 已完成
- [x] 阅读并梳理本地指南中数据格式、训练入口、推理验证方式与潜在需要改动点。
- [x] 拉取并阅读 Issue #4 的 issue 正文（用于约束与范围确认：STT 微调）。
- [x] 增加本地 HF 下载脚本：`scripts/hf_download.py`（HF 缓存固定到 `./.hf_home/`，快照落盘到 `./hf_models/`）。
- [x] 增加数据自检脚本：`scripts/validate_stt_dataset.py`，并用 `data/stt_zh/` 的 smoke 样例自检通过。
- [x] 下载 Kyutai STT 1B 基座到 `./hf_models/kyutai/stt-1b-en_fr-candle/`（见 4.3）。
- [x] 准备最小可跑的数据目录 `data/stt_zh/` + 训练 YAML `moshi-finetune/example/stt_zh_lora.yaml`。

### 进行中
- [ ] 做一次最小 smoke-check（数据管线/脚本参数/形状），确认能启动训练或至少能跑到 dataloader。

## 4. 本地下载 HF 模型（项目内落盘，不写入 `~/.cache`）

已在仓库新增脚本：`scripts/hf_download.py`（会把 `HF_HOME` 指向 `./.hf_home/`，并把快照落到 `./hf_models/`）。

### 4.1 下载 Kyutai STT 1B（基座）
在本仓库根目录执行（推荐用 `uv`，确保 `huggingface_hub` 可用）：
```bash
cd moshi-finetune
uv run python ../scripts/hf_download.py kyutai/stt-1b-en_fr-candle
```

期望产物（示例路径）：
- `hf_models/kyutai/stt-1b-en_fr-candle/model.safetensors`
- `hf_models/kyutai/stt-1b-en_fr-candle/tokenizer_en_fr_audio_8000.model`
- `hf_models/kyutai/stt-1b-en_fr-candle/mimi-pytorch-e351c8d8@125.safetensors`

> 注：脚本会把 HF 相关缓存也固定在项目内（`./.hf_home/`），避免写到用户目录。

### 4.2 当前环境的网络限制（需要你在可联网机器上执行下载）
本工作区的 shell 环境无法解析外网域名（例如 `huggingface.co`），因此**无法在这里实际完成 HF 下载**。  
已实现下载脚本与训练侧“本地 HF_HOME”约束；你只需要在可联网环境运行 4.1 的命令即可把模型落盘到本仓库目录。

### 4.3 你已完成下载的验证（2026-02-03）
已在本仓库检测到：
- `hf_models/kyutai/stt-1b-en_fr-candle/model.safetensors`（约 1.98GB）
- `hf_models/kyutai/stt-1b-en_fr-candle/mimi-pytorch-e351c8d8@125.safetensors`（约 0.38GB）
- `hf_models/kyutai/stt-1b-en_fr-candle/tokenizer_en_fr_audio_8000.model`
- `hf_models/kyutai/stt-1b-en_fr-candle/config.json`

## 5. 训练配置（LoRA：先跑通）

已新增最小训练配置：
- `moshi-finetune/example/stt_zh_lora.yaml`
- `moshi-finetune/example/stt_zh_lora_local.yaml`（完全使用本地 `hf_models/` 文件，避免任何 HF Hub 访问）

已新增数据目录说明：
- `data/stt_zh/README.md`

### 5.1 为 STT 微调新增/暴露的关键选项
- `moshi-finetune/finetune/args.py`：新增 `interleaver.audio_delay_sec`、`interleaver.downmix_to_mono` 等字段（可在 YAML 中配置）。
- `moshi-finetune/train.py`：将上述字段传入 `Interleaver(...)` / `InterleavedTokenizer(...)`，并默认把 HF 缓存固定到项目内 `./.hf_home/`。
- `moshi-finetune/finetune/data/interleaver.py`：训练侧对多声道 wav 进行 downmix（当 `downmix_to_mono: true`）。
- `scripts/stt_from_file_pytorch.py`：推理侧对多声道音频自动 downmix，并默认把 HF 缓存固定到项目内 `./.hf_home/`。

启动训练（单卡）：
```bash
cd moshi-finetune
uv run torchrun --nproc-per-node 1 -m train example/stt_zh_lora.yaml
```

推理验证（合并权重 `save_adapters: false` 的 checkpoint）：
```bash
uv run ../scripts/stt_from_file_pytorch.py \
  --hf-repo kyutai/stt-1b-en_fr-candle \
  --moshi-weight ../runs/stt_zh_lora_run1/checkpoints/checkpoint_000100/consolidated/consolidated.safetensors \
  /abs/path/to/test.wav
```

## 6. 数据集自检（建议训练前先跑）

已新增轻量校验脚本（stdlib-only）：
- `scripts/validate_stt_dataset.py`

示例：
```bash
python scripts/validate_stt_dataset.py data/stt_zh/train.jsonl
```

## 7. 当前阻塞点（需要你在本机终端执行/确认）

### 7.1 Python 依赖尚未安装（当前直接运行会报缺包）
在当前环境直接执行会报：
- `python scripts/stt_from_file_pytorch.py --help` -> `ModuleNotFoundError: julius`
- `python moshi-finetune/train.py --help` -> `ModuleNotFoundError: fire`

建议用 `uv` 或 `pip install -e` 把依赖装好（见 `moshi-finetune/README.md` 的安装段落）。

### 7.2 CUDA 运行时在本工作区不可用（Error 304）
虽然 `nvidia-smi` 可以识别到 GPU，但在本工作区内：
- `torch.cuda.is_available()` 为 `False`，并报 `Error 304: OS call failed or operation not supported on this OS`
- 使用 CUDA runtime 直接调用 `cudaGetDeviceCount` 也返回同样的 `Error 304`

这会导致训练脚本无法在此工作区实际启动 GPU 训练。  
请你在**你的本机终端**确认 WSL/驱动/环境是否允许 CUDA 计算；若仅 Codex 运行环境受限，则你本机直接运行训练命令应可正常使用 GPU。

## 8. 下一步（你执行，我根据输出继续调参/排错）

### 8.1 安装训练依赖（推荐单独虚拟环境）
在仓库根目录：
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e moshi-finetune
```

### 8.2 CUDA 可用性快速自检
```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

### 8.3 跑一个最小训练 smoke run（建议先 5~20 steps）
先把 `moshi-finetune/example/stt_zh_lora_local.yaml` 里的：
- `data.train_data`
- `run_dir`
确认无误，然后执行：
```bash
cd moshi-finetune
torchrun --nproc-per-node 1 -m train example/stt_zh_lora_local.yaml
```

> 如果你希望只用 HF repo id（而不是本地路径），可以改用 `example/stt_zh_lora.yaml`。

### 8.4 推理验证（用合并权重的 checkpoint）
训练跑出第一个 checkpoint 后，把路径替换成你实际的 checkpoint：
```bash
python scripts/stt_from_file_pytorch.py \
  --hf-repo kyutai/stt-1b-en_fr-candle \
  --moshi-weight runs/stt_zh_lora_run1/checkpoints/checkpoint_000100/consolidated/consolidated.safetensors \
  data/stt_zh/wavs/smoke_000001.wav
```

把以下输出贴给我（越完整越好）：
- `torch.cuda.is_available()` 的结果
- 训练启动日志（从加载模型到第一个 step）
- 第一条推理结果（含时间戳）


## 3. 关键决策 / 待确认信息（会影响 offset、分词与评测）
- [ ] 时间戳粒度：字 / 词 / 句子片段？
- [ ] 音频形式：短 utterance（单句）还是长录音（多句连续，需要切片）？
- [ ] 音频声道：是否普遍为 stereo？（若是，训练与推理都必须 downmix 到 mono）
- [ ] 期望输出时间戳：字级 / 词级 / 句级？

> 备注：DSM 固定延迟（`asr_delay_in_tokens`）可能导致整体时间戳偏移；后续将记录偏移量与选择的数据侧/训练侧对齐方案。
