# WebDataset 使用指南

## 概述

WebDataset 格式用于高效处理海量小文件（数十亿级别音频文件）。它将多个小文件打包成 tar 归档，避免了文件系统瓶颈。

**适用场景**:
- ✅ 训练数据规模 > 100 万条音频
- ✅ 音频文件数量导致 inode 耗尽
- ✅ 分布式训练需要高效数据加载
- ✅ 需要从网络存储 (NFS/HDFS) 加载数据

**相比传统 JSONL 格式的优势**:
- **文件系统友好**: 避免海量小文件导致的 inode 和目录查找瓶颈
- **网络传输优化**: tar 文件顺序读取，减少网络往返次数
- **分布式训练友好**: 每个 shard 可独立分配给不同节点
- **内存效率**: 流式解码，不需要一次性加载所有元数据

## 快速开始

### 1. 转换数据为 WebDataset

```bash
# 激活 conda 环境
conda activate ala

# 转换数据
python scripts/convert_to_webdataset.py \
    --input_dir data/stt_zh \
    --output_dir data/stt_zh_webdataset \
    --samples_per_shard 1000
```

参数说明:
- `--input_dir`: 输入目录，包含 `train.jsonl` 和 `wavs/` 目录
- `--output_dir`: 输出目录，生成的 tar 文件存放位置
- `--samples_per_shard`: 每个 tar 文件包含的样本数（推荐 1000-10000）

### 2. 在训练中使用

修改训练配置 YAML:

```yaml
# moshi-finetune/example/stt_zh_lora.yaml
data:
  train_data: "/path/to/data/stt_zh_webdataset"
  use_webdataset: true
  shuffle: true

# 其他配置...
```

启动训练:
```bash
cd moshi-finetune
torchrun --nproc-per-node 2 -m train example/stt_zh_lora.yaml
```

## 数据格式详解

### 输入格式要求

输入目录结构:
```
data/stt_zh/
├── train.jsonl              # 训练清单
└── wavs/                    # 音频目录
    ├── sample_001.wav
    ├── sample_001.json      # 对齐标注
    ├── sample_002.wav
    ├── sample_002.json
    └── ...
```

**train.jsonl** 格式（每行一条）:
```json
{"path": "wavs/sample_001.wav", "duration": 12.34}
```

**sample_001.json** 格式（对齐标注）:
```json
{
  "alignments": [
    ["你", [0.12, 0.18], "SPEAKER_MAIN"],
    ["好", [0.18, 0.24], "SPEAKER_MAIN"],
    ["，", [0.24, 0.30], "SPEAKER_MAIN"]
  ]
}
```

字段说明:
- `alignments`: 词级时间对齐列表
  - 第一项: 文本片段（词/字）
  - 第二项: [开始时间(秒), 结束时间(秒)]
  - 第三项: 说话人标识（通常为 `"SPEAKER_MAIN"`）

### 输出格式

转换后生成多个 tar 文件:
```
data/stt_zh_webdataset/
├── shard-000000.tar
├── shard-000001.tar
├── shard-000002.tar
└── ...
```

每个 tar 文件内部:
```
sample_000000.json  # 元数据（alignments, duration, path）
sample_000000.wav   # 音频文件
sample_000001.json
sample_000001.wav
...
```

## 检查和验证

### 基本统计信息

```bash
# 查看数据集统计信息
python scripts/inspect_webdataset.py --data_dir data/stt_zh_webdataset
```

输出示例:
```
======================================================================
WebDataset Inspector
======================================================================

Data directory: data/stt_zh_webdataset
Number of shards: 100

Shard files:
  shard-000000.tar     45.23 MB
  shard-000001.tar     44.89 MB
  ...
  Total               4523.45 MB

Scanning samples...
Total samples: 100000
Total duration: 12345.67 seconds (3.43 hours)
Average duration: 0.12 seconds
```

### 显示样本详情

```bash
# 显示前 5 个样本的详细信息
python scripts/inspect_webdataset.py \
    --data_dir data/stt_zh_webdataset \
    --show_samples 5
```

### 音频统计信息

```bash
# 显示音频统计（采样率、时长分布等）
python scripts/inspect_webdataset.py \
    --data_dir data/stt_zh_webdataset \
    --audio_stats
```

### 验证数据完整性

```bash
# 验证所有样本可以正确解码
python scripts/inspect_webdataset.py \
    --data_dir data/stt_zh_webdataset \
    --verify
```

### 导出样本列表

```bash
# 导出所有样本 key 到文件
python scripts/inspect_webdataset.py \
    --data_dir data/stt_zh_webdataset \
    --export_keys keys.txt
```

## Python API

### 基本遍历

在代码中遍历 webdataset:

```python
from scripts.inspect_webdataset import iterate_webdataset

# 遍历数据集
for sample in iterate_webdataset("data/stt_zh_webdataset"):
    key = sample["key"]           # 样本 key
    metadata = sample["json"]     # 元数据（alignments, duration 等）
    audio = sample["audio_data"]  # numpy 数组 (samples,)
    sr = sample["sample_rate"]    # 采样率 (24000)

    print(f"{key}: {metadata['duration']:.2f}s")

    # 使用 alignments
    for word, (start, end), speaker in metadata["alignments"]:
        print(f"  {word}: {start:.2f}s - {end:.2f}s")
```

### 指定 shard 范围

使用 brace expansion 模式:

```python
# 指定特定 shard 范围
for sample in iterate_webdataset(
    "data/stt_zh_webdataset",
    pattern="shard-{000000..000100}.tar"
):
    process(sample)
```

### 训练数据加载器

在 `moshi-finetune` 中的实际使用:

```python
from finetune.data.webdataset_loader import build_webdataset_loader

# 创建数据加载器
loader = build_webdataset_loader(
    data_path="data/stt_zh_webdataset",
    instruct_tokenizer=tokenizer,
    batch_size=4,
    rank=0,
    world_size=1,
    shuffle=True,
    shuffle_buffer=1000,
)

# 迭代批次
for batch in loader:
    codes = batch.codes      # [B, num_codebooks, T]
    mask = batch.mask        # [B, T]
    # 训练...
```

## 性能调优

### Shard 大小选择

- **小数据集 (< 10万样本)**: `samples_per_shard=1000`
- **中等数据集 (10万-1000万)**: `samples_per_shard=5000`
- **大数据集 (> 1000万)**: `samples_per_shard=10000`

原则: 每个 shard 文件大小控制在 50MB-500MB 之间

### Shuffle Buffer

训练时的 shuffle buffer 大小:
- **单机训练**: `shuffle_buffer=1000-5000`
- **多机训练**: `shuffle_buffer=100-1000`（每个 worker）

### 音频预处理建议

转换前预处理音频可提升训练性能:

```bash
# 统一采样率到 24kHz, 单声道
for wav in wavs/*.wav; do
    ffmpeg -i "$wav" -ar 24000 -ac 1 "${wav%.wav}_24k_mono.wav"
done
```

## 故障排除

### 问题: 找不到 tar 文件

**症状**:
```
ValueError: No .tar files found in data/stt_zh_webdataset
```

**解决**:
```bash
# 检查输出目录
ls data/stt_zh_webdataset/

# 确认 tar 文件存在
find data/stt_zh_webdataset -name "*.tar"
```

### 问题: 样本数量不匹配

**症状**: 转换后样本数量少于预期

**排查**:
```bash
# 检查源数据
wc -l data/stt_zh/train.jsonl

# 检查转换日志
python scripts/convert_to_webdataset.py \
    --input_dir data/stt_zh \
    --output_dir data/stt_zh_webdataset_debug \
    --samples_per_shard 1000 2>&1 | tee convert.log

# 查看警告信息
grep "WARNING\|not found" convert.log
```

### 问题: 训练时解码失败

**症状**:
```
Failed to decode sample: ...
```

**解决**:
```bash
# 验证所有样本
python scripts/inspect_webdataset.py \
    --data_dir data/stt_zh_webdataset \
    --verify

# 找出有问题的样本
python scripts/inspect_webdataset.py \
    --data_dir data/stt_zh_webdataset \
    --verify 2>&1 | grep "ERROR"
```

## 高级用法

### 多个 JSONL 文件

如果输入目录包含多个 jsonl 文件:

```
data/stt_zh/
├── train_part1.jsonl
├── train_part2.jsonl
├── wavs/
└── ...
```

转换脚本会为每个 jsonl 创建单独的 shard 子目录:

```
data/stt_zh_webdataset/
├── train_part1/
│   ├── shard-000000.tar
│   └── ...
└── train_part2/
    ├── shard-000000.tar
    └── ...
```

训练时指定父目录即可:
```yaml
data:
  train_data: "data/stt_zh_webdataset"
```

### Brace Expansion 模式

在训练配置中使用 brace expansion:

```yaml
data:
  # 只使用前 100 个 shard
  train_data: "data/stt_zh_webdataset/shard-{000000..000099}.tar"
```

## 与传统 JSONL 的对比

| 特性 | JSONL + JSON 文件 | WebDataset (tar) |
|------|------------------|------------------|
| 适用规模 | < 100万样本 | > 100万样本 |
| 文件系统压力 | 高 (每样本2个文件) | 低 (tar 打包) |
| 随机访问 | 支持 | 不支持 |
| 顺序读取速度 | 慢 (大量 open/close) | 快 (流式读取) |
| 网络存储友好度 | 差 | 优 |
| 实现复杂度 | 简单 | 中等 |
| 存储效率 | 中等 | 高 (减少元数据) |

**使用建议**:
- 小数据集 (< 100万): 使用 JSONL 格式即可
- 大数据集 (> 100万): 强烈建议使用 WebDataset
- 超大数据集 (> 1亿): 必须使用 WebDataset

## 相关脚本

| 脚本 | 功能 | 位置 |
|-----|------|-----|
| `convert_to_webdataset.py` | 转换 JSONL 到 webdataset | `scripts/` |
| `inspect_webdataset.py` | 检查和验证 webdataset | `scripts/` |
| `webdataset_loader.py` | 训练数据加载器实现 | `moshi-finetune/finetune/data/` |

## 参考资料

- [WebDataset 官方文档](https://github.com/webdataset/webdataset)
- [Kyutai STT 微调指南](MoshiFinetune_微调Kyutai_STT_中文数据_调研与操作指南.md)
- [Docker 训练指南](DOCKER_FINETUNE_GUIDE.md)
