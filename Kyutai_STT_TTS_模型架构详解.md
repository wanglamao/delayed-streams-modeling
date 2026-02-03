# Kyutai STT & TTS 模型架构详解

## 目录
1. [概述](#概述)
2. [STT (语音转文本) 模型架构](#stt-模型架构)
3. [TTS (文本转语音) 模型架构](#tts-模型架构)
4. [关键技术细节](#关键技术细节)

---

## 概述

本代码库实现了基于 **Delayed Streams Modeling (DSM)** 的流式语音处理模型:
- **Kyutai STT**: 实时语音识别,支持英语和法语
- **Kyutai TTS**: 高质量语音合成

核心技术论文: [Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling](https://arxiv.org/abs/2509.08753)

---

## STT 模型架构

### 整体架构

STT模型采用 **Delayed Streams Modeling** 方法,包含以下核心组件:

```
音频输入 → Mimi编码器 → LM模型 → 文本输出
```

---

### 1. Mimi 音频编码器 (压缩模型)

**文件位置**: `moshi/models/compression.py`

#### 1.1 SEANet 编码器

**输入**:
- 原始音频波形: `[B, 1, T_audio]`
- 采样率: 24000 Hz

**架构配置** (来自 `loaders.py:38-57`):
```python
channels: 1
dimension: 512
n_filters: 64
n_residual_layers: 1
ratios: [8, 6, 5, 4]  # 下采样因子
kernel_size: 7
activation: ELU
causal: True
```

**处理流程**:
1. 输入音频 `[B, 1, T_audio]`
2. 通过4层卷积下采样 (8×6×5×4 = 960倍)
3. 每层包含残差块和激活函数

**输出**:
- 潜在表示: `[B, 512, T_latent]`
- 其中 `T_latent = T_audio / 960`

