# WebDataset 实施状态

## 概述

WebDataset 支持已完成开发和测试，可用于 Kyutai STT 的大规模数据微调训练。

## 完成日期

2026-02-06

## 实现内容

### 1. 核心功能

- ✅ 数据转换脚本 (`scripts/convert_to_webdataset.py`)
  - 支持 JSONL + WAV 文件转换为 tar 格式
  - 支持多个 JSONL 文件批量转换
  - 自动生成 manifest 文件
  - 完整的错误处理和日志

- ✅ 数据检查脚本 (`scripts/inspect_webdataset.py`)
  - 显示数据集统计信息
  - 验证数据完整性
  - 显示样本详情
  - 导出样本列表
  - 音频统计分析

- ✅ 训练数据加载器 (`moshi-finetune/finetune/data/webdataset_loader.py`)
  - 流式解码 tar 文件
  - 支持分布式训练 (DDP)
  - 支持 shuffle 和 buffer
  - 自动音频分块和对齐处理
  - 完整的错误处理

### 2. 文档

- ✅ 完整使用指南 (`WEBDATASET_GUIDE.md`)
  - 快速开始教程
  - 数据格式详解
  - 性能调优建议
  - 故障排除指南
  - Python API 文档

- ✅ 集成到现有文档
  - 主 CLAUDE.md 添加 WebDataset 章节
  - 微调操作指南添加 WebDataset 选项
  - Docker 指南提及 WebDataset 支持
  - 进展记录更新完成状态

### 3. 测试验证

- ✅ 数据转换测试
  - 小规模数据集转换成功
  - 生成的 tar 文件结构正确

- ✅ 训练集成测试
  - 与 moshi-finetune 集成成功
  - 可正常启动训练
  - 数据加载正常

## 适用场景

### 推荐使用 WebDataset (> 100万样本)

- 训练数据超过 100 万条音频
- 文件系统 inode 不足
- 使用网络存储 (NFS/HDFS)
- 多机分布式训练

### 继续使用 JSONL (< 100万样本)

- 小规模数据集 (< 100 万条)
- 本地 SSD 存储
- 需要频繁修改数据

## 性能指标

与传统 JSONL 格式对比:

| 指标 | JSONL | WebDataset | 提升 |
|------|-------|-----------|------|
| 文件系统压力 | 高 | 低 | ✓✓✓ |
| 顺序读取速度 | 慢 | 快 | ✓✓ |
| 存储效率 | 中等 | 高 | ✓✓ |
| 网络友好度 | 差 | 优 | ✓✓✓ |

## 使用示例

### 转换数据

```bash
conda activate ala
python scripts/convert_to_webdataset.py \
  --input_dir data/stt_zh \
  --output_dir data/stt_zh_webdataset \
  --samples_per_shard 5000
```

### 检查数据

```bash
python scripts/inspect_webdataset.py \
  --data_dir data/stt_zh_webdataset \
  --show_samples 10 \
  --audio_stats
```

### 训练配置

```yaml
# moshi-finetune/example/stt_zh_lora.yaml
data:
  train_data: "data/stt_zh_webdataset"
  use_webdataset: true
  shuffle: true
```

### 启动训练

```bash
cd moshi-finetune
torchrun --nproc-per-node 2 -m train example/stt_zh_lora.yaml
```

## 相关文件

| 文件 | 说明 |
|------|------|
| `WEBDATASET_GUIDE.md` | 完整使用文档 |
| `scripts/convert_to_webdataset.py` | 数据转换脚本 |
| `scripts/inspect_webdataset.py` | 数据检查脚本 |
| `moshi-finetune/finetune/data/webdataset_loader.py` | 训练加载器 |

## 后续优化 (可选)

以下优化可根据实际需求添加:

- [ ] 支持增量转换 (避免重复转换已处理的数据)
- [ ] 支持音频压缩 (FLAC/Opus) 减少存储
- [ ] 添加数据统计缓存 (加速 inspect 速度)
- [ ] 支持从 S3/OSS 直接读取 tar 文件

## 参考资料

- [WebDataset 官方文档](https://github.com/webdataset/webdataset)
- [Kyutai STT 微调指南](MoshiFinetune_微调Kyutai_STT_中文数据_调研与操作指南.md)
- [训练进展记录](KYUTAI_STT1B_中文微调_进展记录.md)
