# Kyutai STT ASR 模型模块详解

本文档详细描述了 Kyutai STT (Speech-To-Text) 语音识别模型的各个模块及其输入输出规范。

## 模型概述

Kyutai STT 是基于 **Delayed Streams Modeling (DSM)** 的流式语音识别系统，包含两个主要版本：
- **kyutai/stt-1b-en_fr**: 10亿参数，英法双语，0.5秒延迟，带语义VAD
- **kyutai/stt-2.6b-en**: 26亿参数，英语专用，2.5秒延迟

---

## 1. Mimi 音频编码器 (Audio Tokenizer)

### 功能说明
将原始音频波形压缩为离散 token 表示。

### 输入规范
| 属性 | 值 |
|------|-----|
| 格式 | 原始 PCM 音频 |
| 采样率 | 24,000 Hz |
| 输入形状 | `[Batch, 1, T_audio]` |
| 块大小 | 1920 个样本 (80ms @ 24kHz) |

### 架构组件

#### SEANet 编码器配置
```toml
channels: 1
dimension: 512
n_filters: 64
n_residual_layers: 1
ratios: [8, 6, 5, 4]  # 下采样因子
kernel_size: 7
activation: ELU
causal: True
```

#### 残差向量量化 (RVQ)
- **码本数量**: 32
- **每个码本词汇量**: 2048
- **量化方法**: 分层残差量化

### 输出规范
| 属性 | 值 |
|------|-----|
| 形状 | `[Batch, 32, T_frames]` |
| 帧率 | 12.5 fps (每帧80ms) |
| Token 范围 | 0-2048 (每个码本) |
| 压缩比 | 960:1 |

### 处理流程
```
原始音频 [B, 1, 1920]
    ↓
4层卷积下采样 (8×6×5×4 = 960×)
    ↓
潜在表示 [B, 512, T_latent]
    ↓
32码本残差向量量化
    ↓
音频 Token [B, 32, 1]
```

---

## 2. Transformer 语言模型 (LM Model)

### 功能说明
处理音频 token 并生成文本预测。

### 模型配置

#### 1B 模型配置
```toml
d_model = 2048              # 隐藏层维度
num_heads = 16              # 注意力头数
num_layers = 16             # Transformer 层数
dim_feedforward = 8192      # FFN 维度 (4× d_model)
causal = true               # 因果注意力
norm_first = true           # 前置归一化
bias_ff = false             # FFN 无偏置
bias_attn = false           # 注意力无偏置
context = 750               # 上下文窗口 (60秒)
max_period = 100000         # RoPE 最大周期
gating = "silu"             # SiLU 激活函数
norm = "RmsNorm"            # RMS 归一化
positional_embedding = "Rope"  # 旋转位置编码
```

#### 2.6B 模型配置
```toml
d_model = 2048
num_heads = 32              # 更多注意力头
num_layers = 48             # 更深 (3倍层数)
dim_feedforward = 8192
context = 375               # 更短上下文 (30秒)
```

### 输入规范
| 属性 | 值 |
|------|-----|
| 音频 Token | `[Batch, 32, T]` |
| 前文文本 Token | `[Batch, 1, T-delay]` |
| 上下文窗口 | 750 帧 (1B) / 375 帧 (2.6B) |

### Token 拼接细节

#### 初始 Token 创建
```python
# 在 _get_initial_token() 中
text_token = [B, 1, 1]  # 文本初始 token
audio_token = [B, 32, 1]  # 音频初始 token (32个码本)

# 沿码本维度拼接 (dim=1)
initial_token = torch.cat([text_token, audio_token], dim=1)
# 结果: [B, 33, 1]  (1文本 + 32音频码本)
```

#### 序列拼接 (训练时)
```python
# 在 forward() 中
full_sequence = torch.cat([initial_token, delayed_codes], dim=2)
# [B, 33, 1] + [B, 33, T] → [B, 33, T+1]
```

### 架构组件

#### 2.1 Token 嵌入层与融合机制

#### Token 结构
模型使用多码本结构:
- **文本 token**: 1 个码本 (索引 0)
- **音频 token**: `n_q` 个码本 (索引 1 到 `n_q`)
- **总码本数**: `num_codebooks = n_q + 1`

#### 融合机制: 嵌入相加 (Embedding Addition)

**关键**: 音频和文本 token **不是拼接 (concat)**，而是**嵌入相加**!

```python
# 1. 音频嵌入求和
audio_emb = None
for cb_index in range(num_audio_codebooks):  # 遍历所有音频码本
    emb = audio_embedding[cb_index](
        sequence[:, cb_index + 1]  # 索引+1跳过文本码本
    )
    audio_emb = emb if audio_emb is None else audio_emb + emb

# 2. 文本嵌入
text_emb = text_embedding(sequence[:, 0])  # 文本在索引 0

# 3. 融合: 相加而非拼接!
fused_input = text_emb + audio_emb  # [B, T, d_model]
```

#### 融合流程图
```
音频 Token [B, 32, T]          文本 Token [B, 1, T]
    │                              │
    ├─ 码本0嵌入 ──┐               │
    ├─ 码本1嵌入 ──┤               │
    ├─ 码本2嵌入 ──┼─ 求和 ────────┤
    │    ...      │               │
    └─ 码本31嵌入 ─┘               │
         │                         │
    [B, T, 2048]              [B, T, 2048]
         │                         │
         └──────── 相加 ───────────┘
                   │
            [B, T, 2048] 融合特征
                   │
            Transformer 处理
```

#### 为什么用相加而非拼接?
| 方法 | 维度变化 | 优点 |
|------|----------|------|
| 拼接 | `[B, T, d_model×(K+1)]` | 保留完整信息但维度爆炸 |
| **相加** | `[B, T, d_model]` | 固定维度，计算高效，信息共享 |

#### 2.2 Transformer 层 (16 或 48 层)

**多头自注意力机制**:
| 属性 | 1B 模型 | 2.6B 模型 |
|------|---------|-----------|
| 注意力头数 | 16 | 32 |
| 头维度 | 128 | 64 |
| 位置编码 | RoPE (旋转位置编码) |
| 掩码 | 因果掩码 |
| 偏置 | 无 |

**前馈网络 (FFN)**:
| 属性 | 值 |
|------|-----|
| 架构 | Linear → SiLU → Linear |
| 隐藏层维度 | 8192 (4× d_model) |
| 门控 | SiLU 激活函数 |
| 偏置 | 无 |

**归一化**:
| 属性 | 值 |
|------|-----|
| 类型 | RMSNorm (均方根层归一化) |
| 位置 | 前置归一化 |
| 应用位置 | 注意力前和 FFN 前 |

#### 2.3 位置编码 (RoPE)
| 属性 | 值 |
|------|-----|
| 类型 | 旋转位置嵌入 |
| 最大周期 | 100,000 |
| 特点 | 相对位置编码，支持滑动窗口 |

### 输出规范
| 属性 | 值 |
|------|-----|
| 隐藏状态 | `[Batch, d_model, T]` |
| 维度 | 2048 维向量 |
| 用途 | 输入到文本预测头和 VAD 头 |

### 流式机制

#### 滑动窗口与环形 KV 缓存
```
阶段1: 累积期 (0-750 帧)
- KV 缓存从 0 增长到 750 帧
- 注意力范围: 从第 0 帧到当前帧

阶段2: 滑动窗口 (>750 帧)
- KV 缓存固定在 750 帧
- 新帧加入时丢弃最旧的帧
- 注意力范围: 始终最近 750 帧 (60秒)
```

---

## 3. 文本预测头 (Text Prediction Head)

### 功能说明
从 Transformer 隐藏状态生成文本 token。

### 输入规范
| 属性 | 值 |
|------|-----|
| 隐藏状态 | `[Batch, d_model, T]` |
| 来源 | 最后一层 Transformer 输出 |

### 架构
| 属性 | 值 |
|------|-----|
| 类型 | 线性投影层 |
| 输入维度 | 2048 (d_model) |
| 输出维度 | 8000 (1B) / 4000 (2.6B) |
| 激活函数 | Softmax (概率分布) |

### 输出规范
| 属性 | 值 |
|------|-----|
| Logits | `[Batch, vocab_size, T]` |
| Token | `[Batch, 1, T]` |

### 特殊 Token
| Token ID | 含义 |
|----------|------|
| 0 | 填充结束 (词边界标记) |
| 3 | 填充 token |
| 4-7999/3999 | 实际文本 token (SentencePiece) |

---

## 4. 语义 VAD 头 (Semantic VAD Heads)

### 功能说明
预测多时间跨度的语音活动检测。

### 可用性
仅在 `kyutai/stt-1b-en_fr` 模型中可用。

### 配置
```toml
num_heads = 4    # 4 个预测头
dim = 6          # 每个头的隐藏维度
```

### 输入规范
| 属性 | 值 |
|------|-----|
| 隐藏状态 | `[Batch, d_model, T]` |
| 来源 | 与文本预测共享的 Transformer 输出 |

### 架构

**4 个独立预测头**:
| 头索引 | 预测目标 |
|--------|----------|
| 头 0 | 0.5 秒停顿 |
| 头 1 | 1.0 秒停顿 |
| 头 2 | 2.0 秒停顿 (Unmute 使用) |
| 头 3 | 3.0 秒停顿 |

每个头:
| 属性 | 值 |
|------|-----|
| 线性层 | `d_model (2048) → dim (6)` |
| 激活函数 | Sigmoid |
| 输出 | 概率值 [0, 1] |

### 输出规范
| 属性 | 值 |
|------|-----|
| 形状 | `[4, Batch, 1, T]` |
| 值范围 | 概率 (0.0 到 1.0) |
| 解释 | >0.5: 检测到停顿; ≤0.5: 用户仍在说话 |

### 多任务学习架构
```
共享 Transformer 特征
         │
    ┌────┴────┐
    │         │
文本分支     VAD 分支
    │         │
Linear(2048→8K)  4×Linear(2048→6)
    │         │
Softmax    Sigmoid
    │         │
文本 Token  [P₀.₅, P₁, P₂, P₃]
```

---

## 5. 文本分词器 (SentencePiece Tokenizer)

### 功能说明
在文本和 token ID 之间转换。

### 规格
| 属性 | 1B 模型 | 2.6B 模型 |
|------|---------|-----------|
| 类型 | SentencePiece (unigram) |
| 词汇量 | 8000 | 4000 |
| 语言 | 英语 + 法语 | 英语 |

### 特殊处理
- 将 `▁` (SentencePiece 空格标记) 替换为实际空格

---

## 6. 模块连接与数据流

### 6.1 前向传播流程

```
1. 音频输入 (80ms 块)
   输入: [1, 1, 1920] PCM 采样 @ 24kHz
   ↓

2. Mimi 编码器
   处理: 卷积 + RVQ
   输出: [1, 32, 1] 音频 token
   ↓

3. Token 嵌入与融合
   处理:
   - 32个音频码本嵌入求和 → [1, 2048, 1]
   - 文本嵌入 → [1, 2048, 1]
   - 相加融合 → [1, 2048, 1]
   输出: 融合特征 [1, 2048, 1]
   ↓

4. Transformer 处理
   处理: 16/48 层注意力 + FFN
   - 上下文窗口上的因果自注意力
   - RoPE 位置编码
   - RMSNorm + SiLU 门控
   输出: [1, 2048, 1] 隐藏状态
   ↓

5. 预测头
   ├─ 文本头
   │  处理: Linear(2048→8000) + Softmax
   │  输出: [1, 1, 1] 文本 token
   │
   └─ VAD 头 (如启用)
      处理: 4×Linear(2048→6) + Sigmoid
      输出: [4, 1, 1, 1] 停顿概率
   ↓

6. 后处理
   - 解码文本 token 为单词
   - 检测词边界 (token_id=0)
   - 计算时间戳
   - 检查 VAD 进行话轮转换
```

### 6.2 流式处理流程

```python
for chunk in audio_chunks:  # 每块 = 80ms
    # 1. 编码音频
    codes = mimi.encode(chunk)  # [B, 32, 1]

    # 2. 生成文本 (带延迟)
    tokens = lm_gen.step(codes)  # [B, 1, 1] 或 None

    # 3. 如有效 token 则解码
    if tokens is not None and tokens[0, 0].item() not in [0, 3]:
        text = tokenizer.id_to_piece(tokens[0, 0].item())
        output(text)
```

### 6.3 延迟流建模 (DSM) 与 Delayed Codes 构建

**关键创新**: 预对齐的 token 流，通过延迟机制让模型看到未来音频后再预测文本

#### 延迟配置
| 模型 | asr_delay_in_tokens | 实际延迟 |
|------|---------------------|----------|
| 1B | 6 | 0.48 秒 (6 × 80ms) |
| 2.6B | 32 | 2.56 秒 (32 × 80ms) |

#### Delayed Codes 构建流程

**Step 1: 原始 Codes**
```
音频: [B, 32, T]  (32个码本)
文本: [B, 1, T]   (1个码本)
合并: [B, 33, T]  (沿码本维度拼接)
```

**Step 2: 应用延迟 (`_delay_sequence`)**
```python
def _delay_sequence(delays, tensor, padding):
    # delays: [d0, d1, d2, ..., d32] 每个码本的延迟
    for k, delay in enumerate(delays):
        # 将序列向右滚动 delay 个位置
        line = tensor[:, k].roll(delay, dims=1)
        # 前面补填充
        line[:, :delay] = padding[:, k]
    return torch.stack(outs, dim=1)
```

**Step 3: 延迟效果可视化**
```
时间轴:     t=0    t=1    t=2    t=3    t=4    t=5    t=6    t=7

原始音频:   │ A0   │ A1   │ A2   │ A3   │ A4   │ A5   │ A6   │ A7   │
            └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

延迟音频:   │ pad  │ pad  │ pad  │ pad  │ pad  │ pad  │ A0   │ A1   │  (延迟6帧)
            └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

原始文本:   │ T0   │ T1   │ T2   │ T3   │ T4   │ T5   │ T6   │ T7   │
            └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

延迟对齐:   │      │      │      │      │      │      │ T0   │ T1   │
            └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
            ↑模型在预测T0时，已经看到了A0到A6的未来音频信息
```

**Step 4: 添加初始 Token**
```python
initial_token = [B, 33, 1]  # 特殊的起始token
delayed_codes = torch.cat([initial_token, delayed_codes], dim=2)
# 最终: [B, 33, T+1]
```

#### 延迟值分配
```python
# 典型配置 (1B模型)
delays = [0, 6, 6, 6, ..., 6]  # 共33个值
# - delays[0] = 0: 文本码本无延迟
# - delays[1:33] = 6: 32个音频码本都延迟6帧

# 实际意义:
# 当模型在位置t预测文本时，它看到的是:
# - 文本: 位置t的token (无延迟)
# - 音频: 位置t-6到t的音频 (有6帧延迟/lookahead)
```

#### 流式推理中的延迟处理
```python
# LMGen 使用环形缓存管理延迟
class LMGen:
    def _step(self, audio_tokens):
        # audio_tokens: [B, 32, 1] 当前帧音频

        # 计算写入位置 (考虑延迟)
        write_pos = (offset + delay) % cache_size

        # 写入缓存
        cache[:, 1:33, write_pos] = audio_tokens

        # 读取时自然对齐
        input_ = cache.gather(...)  # 延迟已通过位置计算处理
```

---

## 7. 输入输出规范汇总

### 7.1 系统级 I/O

**输入**:
| 属性 | 值 |
|------|-----|
| 格式 | 原始音频 (WAV, MP3, OGG 等) |
| 采样率 | 任意 (内部重采样为 24kHz) |
| 通道 | 单声道 (多通道取平均) |
| 处理 | 80ms 块 (1920 采样) |

**输出**:
| 属性 | 值 |
|------|-----|
| 文本 | 流式转录，带词级时间戳 |
| VAD | 实时停顿检测 (4 个时间跨度) |
| 延迟 | 0.5s (1B) / 2.5s (2.6B) |

### 7.2 模块级 I/O 表

| 模块 | 输入形状 | 输入类型 | 输出形状 | 输出类型 |
|------|----------|----------|----------|----------|
| **Mimi 编码器** | [B, 1, 1920] | float32 PCM | [B, 32, 1] | int64 token |
| **Token 拼接** | [B, 1, T] + [B, 32, T] | int64 token | [B, 33, T] | int64 token |
| **嵌入融合** | [B, 33, T] | int64 token | [B, T, 2048] | float32 特征 |
| **Transformer** | [B, T, 2048] | float32 特征 | [B, T, 2048] | float32 特征 |
| **文本头** | [B, T, 2048] | float32 特征 | [B, 1, T] | int64 token |
| **VAD 头** | [B, T, 2048] | float32 特征 | [4, B, 1, T] | float32 概率 |
| **分词器** | [T] | int64 token | string | 文本 |

---

## 8. 性能特征

### 8.1 模型对比

| 特性 | 1B 模型 | 2.6B 模型 |
|------|---------|-----------|
| 参数量 | ~10亿 | ~26亿 |
| 语言 | 英语 + 法语 | 英语 |
| 延迟 | 0.5 秒 | 2.5 秒 |
| 上下文窗口 | 60 秒 (750 帧) | 30 秒 (375 帧) |
| 语义 VAD | ✓ 有 | ✗ 无 |
| 层数 | 16 | 48 |
| 注意力头数 | 16 | 32 |
| 批大小 (L40S) | 64 | 16 |
| 实时因子 (H100) | 400× | 400× |
| 内存使用 | ~2GB | ~5GB |

### 8.2 基准测试结果 (2.6B 模型 @ H100)

| 数据集 | CER | WER | 语料库 WER | RTF |
|--------|-----|-----|------------|-----|
| LibriSpeech Clean | 0.67% | 1.95% | 1.69% | 68.19× |
| LibriSpeech Other | 2.31% | 5.24% | 4.33% | 44.76× |
| Meanwhile | 2.02% | 5.50% | 5.60% | 69.19× |
| Tedlium | 2.15% | 3.65% | 3.33% | 67.44× |

---

## 9. 关键文件位置

### Python 脚本
- `scripts/stt_from_file_pytorch.py` - 流式 STT 演示
- `scripts/stt_from_file_with_prompt_pytorch.py` - 提示功能演示
- `scripts/stt_evaluate_on_dataset.py` - 数据集批量评估

### Rust 实现
- `stt-rs/src/main.rs` - 生产级 Rust 实现

### 配置文件
- `configs/config-stt-en_fr-hf.toml` - 1B 模型配置
- `configs/config-stt-en-hf.toml` - 2.6B 模型配置

---


## 11. 音频与文本 Token 融合详解

### 11.1 融合流程概览

```
┌─────────────────────────────────────────────────────────────┐
│  音频编码 (Mimi)              文本编码 (SentencePiece)        │
│  [B, 32, T]                   [B, 1, T]                       │
│       │                            │                        │
│       └────────────┬───────────────┘                        │
│                    │                                        │
│              沿码本维度拼接                                  │
│              torch.cat(..., dim=1)                          │
│                    │                                        │
│              [B, 33, T]                                     │
│                    │                                        │
│       ┌────────────┼────────────┐                          │
│       │            │            │                          │
│   音频嵌入      文本嵌入       延迟处理                      │
│   (32码本求和)   (1码本)       (DSM)                        │
│       │            │            │                          │
│       └────────────┼────────────┘                          │
│                 相加融合                                     │
│              input = text_emb + audio_emb                   │
│                    │                                        │
│              [B, T, 2048]                                   │
│                    │                                        │
│              Transformer 处理                               │
└─────────────────────────────────────────────────────────────┘
```

### 11.2 关键操作详解

#### 1. Token 拼接 (沿码本维度)
```python
# 输入
audio_tokens:  [B, 32, T]  # 32个音频码本
text_tokens:   [B, 1, T]   # 1个文本码本

# 拼接 (dim=1 是码本维度)
combined = torch.cat([text_tokens, audio_tokens], dim=1)
# 输出: [B, 33, T]  (1+32=33个码本)
```

#### 2. 嵌入查找与融合
```python
# 音频嵌入: 32个码本分别嵌入后求和
audio_emb = sum([
    audio_embeddings[i](combined[:, i+1])  # i+1 跳过文本码本
    for i in range(32)
])  # [B, T, 2048]

# 文本嵌入
text_emb = text_embedding(combined[:, 0])  # 索引0是文本码本
# [B, T, 2048]

# 融合: 逐元素相加
fused = text_emb + audio_emb  # [B, T, 2048]
```

#### 3. 延迟流建模 (DSM)
```python
# 配置中的延迟参数
asr_delay_in_tokens = 6  # 1B模型 (0.48秒)
asr_delay_in_tokens = 32 # 2.6B模型 (2.56秒)

# 延迟对齐: 文本输出相对于音频输入有固定延迟
# 音频帧:  [A0, A1, A2, A3, A4, A5, A6, A7, ...]
# 文本帧:  [__, __, __, __, __, __, T0, T1, ...]  (延迟6帧)
```

### 11.3 融合 vs 拼接对比

| 特性 | 拼接 (Concatenation) | 融合 (Addition) |
|------|---------------------|-----------------|
| **操作** | `torch.cat([a, b], dim=-1)` | `a + b` |
| **输出维度** | `[B, T, 4096]` (翻倍) | `[B, T, 2048]` (保持不变) |
| **计算量** | 大 (维度高) | 小 (维度固定) |
| **信息交互** | 后期通过注意力 | 立即融合 |
| **使用场景** | 早期融合 | **Kyutai STT 使用** |

### 11.4 流式处理中的 Token 管理

```python
# LMGen._step() 中的缓存机制
class LMGen:
    def _step(self, audio_tokens):
        # audio_tokens: [B, 32, 1] - 当前帧的音频token

        # 1. 写入缓存 (带延迟)
        cache[:, 1:33, write_position] = audio_tokens  # 音频码本
        cache[:, 0, write_position] = last_text_token   # 文本码本

        # 2. 从缓存读取完整序列
        sequence = cache.gather(...)  # [B, 33, S]

        # 3. 嵌入融合
        fused = self.embed_and_fuse(sequence)  # [B, S, 2048]

        # 4. Transformer 前向
        output = self.transformer(fused)

        # 5. 生成新文本token
        new_text_token = sample(output)
        return new_text_token
```

### 11.5 Delayed Codes 构建详解

#### 构建流程
```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: 编码                                                   │
│  音频: Mimi → [B, 32, T]                                        │
│  文本: SentencePiece → [B, 1, T]                                │
│                                                                 │
│  Step 2: 拼接 (码本维度)                                         │
│  combined = cat([text, audio], dim=1) → [B, 33, T]              │
│                                                                 │
│  Step 3: 应用延迟 (_delay_sequence)                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  for each codebook k:                                   │   │
│  │    delay = delays[k]  # 码本k的延迟值                    │   │
│  │    tensor[:, k] = roll(tensor[:, k], delay)  # 右移      │   │
│  │    tensor[:, k, :delay] = padding  # 前面补填充          │   │
│  └─────────────────────────────────────────────────────────┘   │
│  → [B, 33, T] (延迟后的codes)                                   │
│                                                                 │
│  Step 4: 添加初始token                                          │
│  delayed_codes = cat([initial, delayed_codes], dim=2)           │
│  → [B, 33, T+1]                                                 │
│                                                                 │
│  Step 5: 嵌入融合                                               │
│  - 音频码本(1-32)嵌入求和 → audio_emb [B, T, 2048]              │
│  - 文本码本(0)嵌入 → text_emb [B, T, 2048]                      │
│  - fused = text_emb + audio_emb                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 延迟配置示例
```python
# 1B模型配置
delays = {
    'text': 0,      # 文本码本不延迟
    'audio': 6,     # 音频码本延迟6帧
}

# 实际效果: 预测T0时，模型能看到A0,A1,A2,A3,A4,A5,A6 (未来6帧)
```

### 11.6 总结

Kyutai STT 模型中音频和文本 token 的处理流程:

1. **编码**: 音频通过 Mimi (32码本)，文本通过 SentencePiece (1码本)
2. **拼接**: 沿码本维度拼接为 `[B, 33, T]`
3. **延迟**: 应用 `_delay_sequence` 实现 DSM 对齐
4. **嵌入**: 32个音频码本嵌入后**求和**，文本单独嵌入
5. **融合**: 文本嵌入与音频嵌入**相加**得到 `[B, T, 2048]`
6. **处理**: Transformer 处理融合后的特征
7. **输出**: 生成新的文本 token，循环继续
