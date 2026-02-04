# Kyutai STT Docker Finetune 指南

## 概述

本文档描述如何使用 Docker 和 Docker Compose 进行 Kyutai STT 模型的微调训练。

## 文件结构

```
delayed-streams-modeling/
├── Dockerfile                      # Docker 镜像构建文件
├── docker-compose.yml              # Docker Compose 服务配置
├── configs/
│   └── stt_zh_lora_docker.yaml    # Docker 环境训练配置
├── data/stt_zh/                    # 训练数据目录
│   ├── train.jsonl
│   └── wavs/
├── hf_models/                      # 模型文件目录（挂载）
│   └── stt-1b-en_fr-candle/
└── runs/                           # 训练输出目录（checkpoints & logs）
```

## 快速开始

### 1. 构建 Docker 镜像

```bash
cd delayed-streams-modeling
docker build -t kyutai-stt-finetune:latest .
```

### 2. 运行训练

```bash
# 启动训练服务
docker compose up kyutai-stt-finetune

# 后台运行
docker compose up -d kyutai-stt-finetune

# 查看日志
docker compose logs -f kyutai-stt-finetune
```

### 3. 进入交互式 Shell（调试）

```bash
# 启动交互式 shell 服务
docker compose run --rm kyutai-stt-shell

# 在容器内手动运行训练
torchrun --nproc-per-node 1 -m train \
  /workspace/delayed-streams-modeling/configs/stt_zh_lora_docker.yaml
```

## 配置说明

### 训练配置 (configs/stt_zh_lora_docker.yaml)

| 参数 | 说明 | 值 |
|------|------|-----|
| `data.train_data` | 训练数据路径 | `/workspace/data/emilia_test_25/train.jsonl` |
| `moshi_paths.moshi_path` | 模型权重路径 | `/workspace/hf_models/stt-1b-en_fr-candle/model.safetensors` |
| `moshi_paths.mimi_path` | Mimi codec 路径 | `/workspace/hf_models/stt-1b-en_fr-candle/mimi-pytorch-...safetensors` |
| `moshi_paths.tokenizer_path` | Tokenizer 路径 | `/workspace/hf_models/stt-1b-en_fr-candle/tokenizer_en_fr_audio_8000.model` |
| `moshi_paths.config_path` | 模型配置文件 | `/workspace/hf_models/stt-1b-en_fr-candle/config.json` |
| `run_dir` | 训练输出目录 | `/workspace/runs/stt_zh_lora_run1` |
| `lora.enable` | 启用 LoRA | `true` |
| `lora.rank` | LoRA rank | `64` |
| `batch_size` | 批次大小 | `2` |
| `max_steps` | 最大训练步数 | `2000` |
| `optim.lr` | 学习率 | `2.0e-5` |

### Docker Compose 服务

#### sft (训练服务)
- **用途**: 运行训练任务
- **GPU**: 使用 GPU 2 和 3 (通过 `CUDA_VISIBLE_DEVICES=2,3` 指定)
- **自动启动**: 是
- **命令**: `torchrun --nproc-per-node 2 -m train <config>`

#### kyutai-stt-shell
- **用途**: 交互式调试
- **GPU**: 使用 GPU 2 和 3
- **自动启动**: 否（需手动运行）
- **命令**: `bash`

### 挂载卷

| 宿主机路径 | 容器路径 | 说明 |
|-----------|---------|------|
| `.` | `/workspace/delayed-streams-modeling` | 代码目录 |
| `/nvmedata/ala_storage/ala_data/emilia_test_25` | `/workspace/data/emilia_test_25` | 训练数据（只读） |
| `/nvmedata/ala_storage/stt-1b-en_fr-candle` | `/workspace/hf_models/stt-1b-en_fr-candle` | 模型文件（只读） |
| `./.hf_home` | `/workspace/.hf_home` | HuggingFace 缓存 |
| `./runs` | `/workspace/runs` | 训练输出 |

### GPU 配置说明

**关键配置** (`docker-compose.yaml`):

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 8              # 分配的总 GPU 数量
          capabilities: [gpu]

environment:
  - CUDA_VISIBLE_DEVICES=2,3    # 实际使用的 GPU ID
  - NCCL_DEBUG=INFO
```

- `count: 8` - Docker 分配 8 个 GPU 给容器
- `CUDA_VISIBLE_DEVICES=2,3` - PyTorch 只使用 GPU 2 和 3
- `--nproc-per-node 2` - torchrun 启动 2 个进程

## 常用命令

```bash
# 构建镜像
docker compose build

# 启动训练
docker compose up sft

# 停止训练
docker compose down

# 查看训练日志
docker compose logs -f sft

# 进入交互式 shell
docker compose run --rm kyutai-stt-shell

# 清理容器和卷
docker compose down -v

# 重新构建并启动
docker compose up --build sft
```

## 故障排除

### 1. GPU 不可用

检查 NVIDIA Docker 运行时：
```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### 2. 指定特定 GPU (如只用 GPU 2,3)

配置方式：`count: 8` (总GPU数) + `CUDA_VISIBLE_DEVICES=2,3` (指定使用哪几张)

```yaml
# docker-compose.yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 8          # 分配所有GPU
          capabilities: [gpu]

environment:
  - CUDA_VISIBLE_DEVICES=2,3  # 只使用 GPU 2 和 3
```

注意：`device_ids` 在较新版本的 Docker Compose 中可能不被支持，建议使用 `count` + `CUDA_VISIBLE_DEVICES` 的组合方式。

### 2. 共享内存不足

已配置 `shm_size: '16gb'`，如需更大可调整 docker-compose.yml。

### 3. 模型文件找不到

检查挂载路径：
```bash
ls /nvmedata/ala_storage/stt-1b-en_fr-candle/
```

### 4. 数据路径错误

检查数据目录结构：
```bash
ls /nvmedata/ala_storage/ala_data/emilia_test_25/
# 应包含: train.jsonl 和 wavs/
```

### 5. 权限问题

确保 runs 目录可写：
```bash
mkdir -p runs && chmod 755 runs
```

## 训练输出

训练结果保存在 `./runs/stt_zh_lora_run1/` 目录：

```
runs/stt_zh_lora_run1/
├── checkpoints/          # 模型检查点
│   ├── checkpoint_100.pt
│   ├── checkpoint_200.pt
│   └── ...
├── logs/                 # 训练日志
│   └── train.log
└── config.yaml           # 保存的配置文件
```

## 自定义配置

复制 `configs/stt_zh_lora_docker.yaml` 并修改参数：

```bash
cp configs/stt_zh_lora_docker.yaml configs/my_config.yaml
# 编辑 my_config.yaml
```

然后修改 docker-compose.yml 中的命令：

```yaml
command: >
  torchrun
  --nproc-per-node 1
  -m train
  /workspace/delayed-streams-modeling/configs/my_config.yaml
```

## 多 GPU 训练

修改 `docker-compose.yaml`：

```yaml
# 使用 GPU 0,1,2,3 (4张卡)
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 8
          capabilities: [gpu]

environment:
  - CUDA_VISIBLE_DEVICES=0,1,2,3   # 指定使用 GPU 0-3

command: >
  torchrun
  --nproc-per-node 4               # 4个进程对应4张卡
  -m train
  /workspace/delayed-streams-modeling/configs/stt_zh_lora_docker.yaml
```

注意：`count` 应该 >= 你实际想使用的 GPU 数量，`CUDA_VISIBLE_DEVICES` 精确控制使用哪些 GPU。

## 参考

- [MoshiFinetune 操作指南](MoshiFinetune_微调Kyutai_STT_中文数据_调研与操作指南.md)
- [训练进展记录](KYUTAI_STT1B_中文微调_进展记录.md)
- 原始配置: `moshi-finetune/example/stt_zh_lora_emilia_test_25.yaml`
