# Semantic VAD (Voice Activity Detection) 实现分析

## 概述

Semantic VAD 是 Kyutai STT 1B 模型（`kyutai/stt-1b-en_fr`）的一个特殊功能，用于检测用户何时停止说话。与传统的基于音频能量的 VAD 不同，Semantic VAD 是基于语义理解的，能够更智能地判断说话的停顿。

## 完整流程概览 (Semantic VAD + ASR)

### 端到端数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          音频输入 (24kHz PCM)                            │
│                     例: "Hello world ... how are you"                    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Step 1: 音频分块 (80ms chunks)                        │
│                         1920 samples per chunk                           │
│                                                                           │
│  Chunk 1: [0-80ms]  Chunk 2: [80-160ms]  Chunk 3: [160-240ms] ...      │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              Step 2: Mimi 音频编码器 (Audio Tokenizer)                   │
│                                                                           │
│  输入: PCM 音频块 (1920 samples)                                         │
│  输出: 音频 tokens (32 codebooks × 1 frame)                              │
│  帧率: 12.5 fps (每秒12.5帧)                                             │
│                                                                           │
│  PCM [1920] → Mimi Encoder → Audio Tokens [32, 1]                       │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│           Step 3: Transformer 主干网络 (Language Model)                  │
│                                                                        │
│  架构: 16层 Transformer, 2048维, 16个注意力头                            │
│  输入: 音频 tokens [32, T] (T = 当前时间步)                              │
│  处理: 自回归生成，考虑历史上下文 (context=750帧 = 60秒)                 │
│                                                                           │
│  Audio Tokens → [Embedding] → [16 × Transformer Layers]                 │
│                                      ↓                                    │
│                              Hidden States [2048, T]                     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│  Step 4a: 文本预测头          │  │  Step 4b: VAD 预测头          │
│                              │  │                              │
│  输入: Hidden States         │  │  输入: Hidden States         │
│  输出: 文本 Token (1个)       │  │  输出: 4个停顿概率           │
│                              │  │                              │
│  Linear(2048 → 8000)         │  │  4 × Linear(2048 → 6)        │
│  ↓                           │  │  ↓                           │
│  Softmax                     │  │  Sigmoid                     │
│  ↓                           │  │  ↓                           │
│  Token ID                    │  │  [P(0.5s), P(1s),           │
│  (例: 1234 → "Hello")        │  │   P(2s), P(3s)]             │
│                              │  │  (例: [0.1, 0.2, 0.8, 0.9]) │
└──────────────┬───────────────┘  └──────────────┬───────────────┘
               │                                 │
               ▼                                 ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│  Step 5a: 文本解码            │  │  Step 5b: VAD 判断            │
│                              │  │                              │
│  Token → SentencePiece       │  │  if P(2s) > 0.5:            │
│  1234 → "Hello"              │  │      用户停止说话            │
│  1235 → "▁world"             │  │  else:                      │
│  0    → <end_of_word>        │  │      用户继续说话            │
└──────────────┬───────────────┘  └──────────────┬───────────────┘
               │                                 │
               └────────────┬────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Step 6: 输出消息流                                 │
│                                                                           │
│  每个音频块 (80ms) 产生:                                                  │
│                                                                           │
│  1. Step 消息 (如果启用 VAD):                                             │
│     {                                                                     │
│       "type": "Step",                                                     │
│       "prs": [0.1, 0.2, 0.8, 0.9],  // 4个时间窗口的停顿概率             │
│       "processed_audio_ms": 1280     // 已处理的音频时长                 │
│     }                                                                     │
│                                                                           │
│  2. Word 消息 (当识别出新单词):                                           │
│     {                                                                     │
│       "type": "Word",                                                     │
│       "text": "Hello",                                                    │
│       "start_time": 0.5              // 单词开始时间                      │
│     }                                                                     │
│                                                                           │
│  3. EndWord 消息 (当单词结束):                                            │
│     {                                                                     │
│       "type": "EndWord",                                                  │
│       "stop_time": 0.8               // 单词结束时间                      │
│     }                                                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 时间线示例

以下是一个实际的处理时间线，展示 ASR 和 VAD 如何协同工作：

```
时间轴 (秒):  0.0    0.5    1.0    1.5    2.0    2.5    3.0    3.5    4.0
音频输入:     [Hello world............how are you...................]
              ^^^^^^^^^^^^          ^^^^^^^^^^^^

ASR 输出:
  0.5s: Word("Hello")
  0.8s: EndWord(stop_time=0.8)
  1.0s: Word("world")
  1.3s: EndWord(stop_time=1.3)
  2.5s: Word("how")
  2.7s: EndWord(stop_time=2.7)
  3.0s: Word("are")
  3.2s: EndWord(stop_time=3.2)
  3.5s: Word("you")
  3.8s: EndWord(stop_time=3.8)

VAD 输出 (P(2s) - 2秒窗口):
  0.0-1.3s: 0.1, 0.2, 0.1, 0.3  (说话中)
  1.3-2.5s: 0.3, 0.5, 0.8, 0.9  (检测到停顿!) ← 在 "world" 后
  2.5-4.0s: 0.1, 0.2, 0.1, 0.2  (继续说话)
  4.0s+:    0.4, 0.6, 0.9, 0.95 (检测到停顿!) ← 在 "you" 后
```

### 关键时序特性

1. **实时处理**: 每 80ms 处理一个音频块
2. **延迟**:
   - 1B 模型: 0.5 秒延迟 (配置: `asr_delay_in_tokens = 6`)
   - 2.6B 模型: 2.5 秒延迟
3. **帧率**: 12.5 fps (每秒输出 12.5 个 token)
4. **上下文窗口**: 750 帧 = 60 秒
   - Transformer 可以回看过去 60 秒的音频历史
   - 计算: 750 帧 ÷ 12.5 fps = 60 秒
   - 这使得模型能够理解长时间的语义上下文
   - **滑动窗口机制**: 超过 750 帧时，使用环形 KV 缓存自动丢弃最旧的帧
5. **VAD 响应**: 实时输出，无额外延迟

### 滑动窗口机制详解

当音频流超过 750 帧（60 秒）时，模型使用 **环形 KV 缓存（Ring KV Cache）** 实现滑动窗口：

#### 工作原理

```
阶段 1: 累积阶段 (0-750 帧)
┌────────────────────────────────────────────────────────────┐
│ 时间: 0s ──────────────────────────────────────────→ 60s   │
│ 帧数: 0 ─────────────────────────────────────────→ 750     │
│ KV Cache: [逐步累积所有历史帧]                            │
│ 注意力范围: 从第 0 帧到当前帧                              │
└────────────────────────────────────────────────────────────┘

阶段 2: 滑动窗口阶段 (>750 帧)
┌────────────────────────────────────────────────────────────┐
│ 时间: 60s ──────────────────────────────────────→ 120s     │
│ 帧数: 750 ─────────────────────────────────────→ 1500      │
│                                                             │
│ t=751: KV Cache [1, 2, 3, ..., 749, 750]  ← 丢弃帧 0      │
│ t=752: KV Cache [2, 3, 4, ..., 750, 751]  ← 丢弃帧 1      │
│ t=753: KV Cache [3, 4, 5, ..., 751, 752]  ← 丢弃帧 2      │
│ ...                                                         │
│                                                             │
│ 注意力范围: 始终保持最近 750 帧 (60 秒)                    │
└────────────────────────────────────────────────────────────┘
```

#### 技术实现

**1. 配置参数**

```toml
[modules.asr.model.transformer]
causal = true                    # 因果注意力（只能看过去）
context = 750                    # 滑动窗口大小
max_seq_len = 40960              # RoPE 的最大序列长度
positional_embedding = "Rope"    # 旋转位置编码（相对位置）
```

**2. 流式处理**

从代码中可以看到模型使用 `streaming_forever` 方法进入流式模式：

```python
# scripts/stt_from_file_pytorch.py
self.mimi.streaming_forever(batch_size)
self.lm_gen.streaming_forever(batch_size)
```

在流式模式下，模型内部自动管理 KV cache 的滑动窗口，每次处理新的音频块时：
- 将新帧的 KV 添加到缓存
- 如果超过 750 帧，自动丢弃最旧的帧
- 注意力机制只在最近 750 帧上计算
```

#### 为什么使用 RoPE？

**旋转位置编码（Rotary Position Embedding）** 是滑动窗口的关键：

```
传统绝对位置编码:
  Frame 0: pos=0, Frame 1: pos=1, ..., Frame 750: pos=750
  问题: 当丢弃 Frame 0 后，Frame 1 的位置还是 1，但它现在是"最旧"的帧

RoPE 相对位置编码:
  只关心帧之间的相对距离，不关心绝对位置
  Frame i 和 Frame j 的关系只取决于 (j - i)
  优势: 滑动窗口时不需要重新计算位置编码
```

#### 内存和计算优势

```
不使用滑动窗口:
  - 10 分钟音频 = 7500 帧
  - KV Cache 大小: 7500 × 2048 × 2 (K+V) ≈ 30M 参数
  - 注意力计算: O(7500²) ≈ 56M 次操作

使用滑动窗口 (750 帧):
  - 无论多长音频，KV Cache 固定: 750 × 2048 × 2 ≈ 3M 参数
  - 注意力计算: O(750²) ≈ 562K 次操作
  - 内存节省: 10× (10分钟音频)
  - 计算节省: 100× (10分钟音频)
```

#### 实际效果

处理 5 分钟音频 (3750 帧) 时：

- **前 60 秒 (0-750 帧)**：模型可以看到从开始到现在的所有内容，上下文逐渐增长
- **60-120 秒 (750-1500 帧)**：模型始终看到最近 60 秒。例如在 t=800 时，可以看到 [51-800] 帧，看不到 [0-50] 帧
- **120-300 秒 (1500-3750 帧)**：同样保持 60 秒滑动窗口，对于语音识别已经足够理解语义

#### 对 VAD 的影响

Semantic VAD 基于最近 60 秒的语义理解做出判断，这个时间窗口足以识别：
- 说话风格和节奏
- 停顿模式
- 对话上下文

**示例对比**：

1. **思考停顿**：用户说 "我想要... [停顿 1 秒] ...一杯咖啡"
   - 模型基于前面的语义（"我想要"）判断这是思考停顿
   - VAD 输出: P(2s) = 0.3 (未检测到结束)

2. **真正结束**：用户说 "一杯咖啡，谢谢。[停顿 2 秒]"
   - 模型基于语义（句子完整，有结束词"谢谢"）
   - VAD 输出: P(2s) = 0.8 (检测到结束)

### 核心组件交互

```
┌─────────────────────────────────────────────────────────────────┐
│                        Kyutai STT 系统                           │
│                                                                   │
│  ┌─────────────┐      ┌──────────────────┐      ┌────────────┐ │
│  │   Mimi      │      │   Transformer    │      │  Output    │ │
│  │  Encoder    │─────▶│   Backbone       │─────▶│  Heads     │ │
│  │             │      │                  │      │            │ │
│  │ 24kHz→12.5fps│     │  16 Layers       │      │ ┌────────┐ │ │
│  │ PCM→Tokens  │      │  2048 dim        │      │ │ Text   │ │ │
│  └─────────────┘      │  Context: 750帧  │      │ │ Head   │ │ │
│                       │  (60秒历史)      │      │ └────┬───┘ │ │
│                       │  Causal Attn     │      │      │     │ │
│                       │  + RoPE          │      │ ┌────▼───┐ │ │
│                       │  + RMSNorm       │      │ │  VAD   │ │ │
│                       │  + SiLU Gating   │      │ │ Heads  │ │ │
│                       └──────────────────┘      │ │ (4个)  │ │ │
│                                                  │ └────────┘ │ │
│                                                  └────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 多任务学习架构

Semantic VAD 和 ASR 共享底层特征，实现高效的多任务学习：

```
                    Shared Transformer Features
                              │
                    ┌─────────┴─────────┐
                    │                   │
              Text Branch          VAD Branch
                    │                   │
            ┌───────▼────────┐   ┌─────▼──────┐
            │ Linear(2048→8K)│   │ 4×Linear   │
            │                │   │ (2048→6)   │
            └───────┬────────┘   └─────┬──────┘
                    │                   │
            ┌───────▼────────┐   ┌─────▼──────┐
            │   Softmax      │   │  Sigmoid   │
            └───────┬────────┘   └─────┬──────┘
                    │                   │
                    ▼                   ▼
              Text Token          [P₀.₅, P₁, P₂, P₃]
            (8000 classes)        (4 pause probs)

优势:
✓ 共享特征提取 → 计算高效
✓ 语义理解 → VAD 更准确
✓ 联合训练 → 互相增强
✓ 单次前向传播 → 无额外延迟
```

### 实际应用流程

#### 场景 1: 语音助手 (Unmute)

```python
# 1. 初始化连接
websocket = connect_to_server("ws://localhost:8080/api/asr-streaming")

# 2. 流式发送音频
while recording:
    audio_chunk = microphone.read(1920)  # 80ms
    send_message({"type": "Audio", "pcm": audio_chunk})

# 3. 接收并处理响应
async for message in websocket:
    if message["type"] == "Step":
        # 检查 VAD (2秒窗口)
        if message["prs"][2] > 0.5:
            # 用户停止说话，触发助手响应
            assistant.start_response()

    elif message["type"] == "Word":
        # 实时显示识别的文字
        display_text(message["text"])
```

#### 场景 2: 转录应用

```python
# 1. 加载模型
lm = load_model("kyutai/stt-1b-en_fr")
lm_gen = LMGen(lm, temp=0)

# 2. 处理音频文件
with lm_gen.streaming(1):
    for audio_chunk in audio_file:
        audio_tokens = mimi.encode(audio_chunk)

        # 同时获取文本和 VAD
        text_tokens, vad_heads = lm_gen.step_with_extra_heads(audio_tokens)

        # 处理文本
        if text_tokens[0, 0, 0] not in (0, 3):
            text = tokenizer.decode(text_tokens)
            transcript.append(text)

        # 使用 VAD 分段
        if vad_heads[2][0, 0, 0] > 0.5:
            segments.append(transcript)
            transcript = []
```

### 性能特性

| 指标 | 1B 模型 (带 VAD) | 2.6B 模型 (无 VAD) |
|------|------------------|-------------------|
| 延迟 | 0.5 秒 | 2.5 秒 |
| 帧率 | 12.5 fps | 12.5 fps |
| VAD 开销 | ~0% (共享计算) | N/A |
| 实时因子 (H100) | 400× | 400× |
| 实时因子 (L40S) | 64× @ 3× | 64× @ 3× |
| 内存占用 | ~2GB | ~5GB |

## 架构设计

### 1. 与 ASR 模型的关系

Semantic VAD **不是一个独立的模型**，而是作为 ASR 模型的**额外预测头（extra heads）**集成在主模型中：

```toml
# configs/config-stt-en_fr-hf.toml
[modules.asr.model.extra_heads]
num_heads = 4    # 4个预测头，对应不同的时间窗口
dim = 6          # 每个头的维度
```

在 Rust 实现中（`stt-rs/src/main.rs:95-102`）：

```rust
let extra_heads = if vad {
    Some(moshi::lm::ExtraHeadsConfig {
        num_heads: 4,  // 4个预测头
        dim: 6,        // 维度为6
    })
} else {
    None
};
```

### 2. 工作原理

#### 多时间窗口预测

Semantic VAD 使用 **4 个预测头**，每个头预测不同长度的停顿：

- **Head 0**: 0.5 秒停顿
- **Head 1**: 1.0 秒停顿
- **Head 2**: 2.0 秒停顿（Unmute 中使用）
- **Head 3**: 3.0 秒停顿

代码参考（`scripts/stt_from_mic_rust_server.py:21-24`）：

```python
# The VAD has several prediction heads, each of which tries to determine whether there
# has been a pause of a given length. The lengths are 0.5, 1.0, 2.0, and 3.0 seconds.
# Lower indices predict pauses more aggressively. In Unmute, we use 2.0 seconds = index 2.
PAUSE_PREDICTION_HEAD_INDEX = 2
```

#### 预测输出

每个预测头输出一个 **概率值**（0-1 之间），表示在该时间窗口内是否有停顿：
- **> 0.5**: 检测到停顿（用户可能已停止说话）
- **≤ 0.5**: 用户仍在说话

## 实现细节

### 1. PyTorch 实现

在 PyTorch 中（`scripts/stt_from_file_pytorch.py:184-190`）：

```python
if args.vad:
    text_tokens, vad_heads = lm_gen.step_with_extra_heads(audio_tokens)
    if vad_heads:
        pr_vad = vad_heads[2][0, 0, 0].cpu().item()  # 使用2秒窗口
        if pr_vad > 0.5 and not last_print_was_vad:
            print(" [end of turn detected]")
            last_print_was_vad = True
else:
    text_tokens = lm_gen.step(audio_tokens)
```

**关键点**：
- 使用 `step_with_extra_heads()` 方法同时获取文本 tokens 和 VAD 预测
- VAD 预测与 ASR 推理在**同一个前向传播**中完成
- 访问 `vad_heads[2]` 获取 2 秒窗口的预测

### 2. Rust 实现

在 Rust 中（`stt-rs/src/main.rs:192-203`）：

```rust
match asr_msg {
    moshi::asr::AsrMsg::Step { prs, .. } => {
        // prs is the probability of having no voice activity for different time
        // horizons.
        // In kyutai/stt-1b-en_fr-candle, these horizons are 0.5s, 1s, 2s, and 3s.
        if self.vad && prs[2][0] > 0.5 && !printed_eot {
            printed_eot = true;
            if !self.timestamps {
                print!(" <endofturn pr={}>\", prs[2][0]);
            } else {
                println!(\"<endofturn pr={}>\", prs[2][0]);
            }
        }
    }
    // ... 处理其他消息类型
}
```

### 3. WebSocket 服务器实现

在 WebSocket 流式传输中（`scripts/stt_from_mic_rust_server.py:27-45`）：

```python
async def receive_messages(websocket, show_vad: bool = False):
    speech_started = False
    async for message in websocket:
        data = msgpack.unpackb(message, raw=False)

        # The Step message only gets sent if the model has semantic VAD available
        if data["type"] == "Step" and show_vad:
            pause_prediction = data["prs"][PAUSE_PREDICTION_HEAD_INDEX]
            if pause_prediction > 0.5 and speech_started:
                print("| ", end="", flush=True)  # 显示停顿标记
                speech_started = False

        elif data["type"] == "Word":
            print(data["text"], end=" ", flush=True)
            speech_started = True
```

**消息类型**：
- **Step**: 包含 VAD 预测的消息，每个音频帧都会发送
- **Word**: 识别出的单词
- **EndWord**: 单词结束，包含时间戳

## 与 ASR 模型的集成方式

### 1. 模型架构

```
输入音频 → Mimi 编码器 → 音频 tokens
                              ↓
                    Transformer 主干网络
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
              文本预测头           VAD 预测头（4个）
                    ↓                   ↓
              文本 tokens         停顿概率 [0.5s, 1s, 2s, 3s]
```

### 2. 音频 Token 与文本 Token 的对齐机制

这是 Delayed Streams Modeling (DSM) 的核心创新。模型需要处理两个不同速率的 token 流：

#### Token 流特性

**音频 Token 流**：
- 来源：Mimi 编码器
- 帧率：12.5 fps（每秒 12.5 帧）
- 每帧：32 个 codebooks（`audio_codebooks = 32`）
- 特点：**固定速率**，与音频时长严格对应

**文本 Token 流**：
- 来源：Transformer 文本预测头
- 帧率：12.5 fps（与音频同步）
- 每帧：1 个文本 token
- 特点：**可变内容**，一个词可能对应多个 token

#### Delayed Streams Modeling 方法

DSM 通过 **预对齐 + 延迟** 的方式解决对齐问题：

```
时间轴:     t=0    t=1    t=2    t=3    t=4    t=5    t=6    t=7
           ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
音频输入:   │ A0   │ A1   │ A2   │ A3   │ A4   │ A5   │ A6   │ A7   │
           └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
                                 ↓ 延迟 (asr_delay_in_tokens)
           ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
文本输出:   │      │      │      │      │      │      │ T0   │ T1   │
           └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

说明:
- 1B 模型: asr_delay_in_tokens = 6 (0.48秒延迟)
- 2.6B 模型: asr_delay_in_tokens = 32 (2.56秒延迟)
- 文本 token T0 对应音频帧 A0 的内容，但延迟 6/32 帧后输出
```

#### 关键配置参数

```toml
# configs/config-stt-en_fr-hf.toml (1B 模型)
[modules.asr]
asr_delay_in_tokens = 6        # 延迟 6 帧 = 0.48 秒

[modules.asr.model]
audio_codebooks = 32           # 每帧 32 个音频 codebooks
text_in_vocab_size = 8001      # 文本输入词汇表大小
text_out_vocab_size = 8000     # 文本输出词汇表大小
```

#### 对齐的三个层次

**1. 帧级对齐（Frame-level Alignment）**

每个音频帧对应一个文本 token 输出位置：

```
音频帧 0 → 文本位置 0 (延迟后输出)
音频帧 1 → 文本位置 1 (延迟后输出)
音频帧 2 → 文本位置 2 (延迟后输出)
...
```

这种 1:1 的帧对应关系由模型架构保证。

**2. 词级对齐（Word-level Alignment）**

使用特殊 token 标记词边界：

- **end_of_padding (token_id=0)**：标记词边界
- **padding_token (token_id=3)**：填充 token

从代码 `scripts/stt_from_file_pytorch.py:46-49` 可以看到：

```python
# Normally `end_of_padding` tokens indicate word boundaries.
# Everything between them should be a single word;
# the time offset of the those tokens correspond to word start and
# end timestamps (minus silence prefix and audio delay).
```

示例：
```
帧:     0    1    2    3    4    5    6    7    8    9
文本:   H    e    l    l    o    EOP  w    o    r    l
词:     [-------- Hello --------]    [------ wor...

EOP = end_of_padding (词边界标记)
```

**3. 时间戳对齐（Timestamp Alignment）**

从 token 位置计算实际时间戳：

```python
# scripts/stt_from_file_pytorch.py:60-64
def _tstmp(start_position, end_position):
    return (
        max(0, start_position / frame_rate - offset_seconds),
        max(0, end_position / frame_rate - offset_seconds),
    )
```

计算公式：
```
实际时间 = (token位置 / 12.5) - 延迟时间 - 静音前缀时间
```

#### 处理不同词长的策略

**情况 1：一个词对应多个 token**

```
词: "Hello"
Token 序列: [H, e, l, l, o, EOP]
帧分配: 5 个帧 + 1 个边界帧

时间计算:
- 开始时间 = 第一个 token 的帧位置 / 12.5
- 结束时间 = EOP token 的帧位置 / 12.5
```

**情况 2：快速说话，多个词挤在一起**

从代码 `scripts/stt_from_file_pytorch.py:85-86` 可以看到：

```python
# We're in a rare situation where multiple words are so close
# they are not separated by `end_of_padding`.
# We tokenize words one-by-one; each word is assigned with
# as many frames as much tokens it has.
```

处理方式：按 token 数量比例分配帧：
```
Token 序列: [H, e, l, l, o, w, o, r, l, d, EOP]
识别结果: "Hello world" (两个词但没有中间的 EOP)

分配策略:
- "Hello" 有 5 个 token → 分配 5 帧
- "world" 有 5 个 token → 分配 5 帧
```

#### 实际数据流示例

```
输入: 用户说 "Hello world"

Step 1: 音频编码
  t=0: PCM[1920] → Mimi → Audio_Tokens[32, 1]
  t=1: PCM[1920] → Mimi → Audio_Tokens[32, 1]
  ...

Step 2: Transformer 处理（带延迟）
  t=0-5: 累积音频 tokens，还未输出文本
  t=6: 输出 Text_Token[H] (对应 t=0 的音频)
  t=7: 输出 Text_Token[e] (对应 t=1 的音频)
  t=8: 输出 Text_Token[l] (对应 t=2 的音频)
  t=9: 输出 Text_Token[l] (对应 t=3 的音频)
  t=10: 输出 Text_Token[o] (对应 t=4 的音频)
  t=11: 输出 Text_Token[EOP] (词边界)
  t=12: 输出 Text_Token[w] (对应 t=6 的音频)
  ...

Step 3: 后处理
  - 检测到 EOP，识别出词 "Hello"
  - 计算时间戳: start=(6-6)/12.5=0.0s, end=(11-6)/12.5=0.4s
  - 发送 Word 消息: {"type": "Word", "text": "Hello", "start_time": 0.0}
  - 发送 EndWord 消息: {"type": "EndWord", "stop_time": 0.4}
```

#### 优势分析

**1. 简化模型训练**
- 不需要学习对齐（alignment-free）
- 对齐在预处理阶段完成
- 模型只需学习 token 预测

**2. 支持流式推理**
- 固定延迟，可预测
- 每帧独立处理
- 无需等待完整句子

**3. 灵活性**
- 支持任意长度的词
- 自动处理快速/慢速说话
- 通过 EOP token 自然分词

**4. 时间戳精度**
- 帧级精度：80ms (1/12.5)
- 足够用于词级时间戳
- 支持实时应用

### 3. 共享特征

- VAD 预测头与文本预测头**共享相同的 Transformer 特征**
- 这使得 VAD 能够基于**语义理解**而非仅仅音频能量
- 额外的计算开销很小（只增加了 4 个小型预测头）

### 3. 训练方式

虽然代码中没有直接展示训练过程，但从架构可以推断：
- VAD 头与主 ASR 模型**联合训练**
- 使用标注的停顿数据作为监督信号
- 多任务学习：同时优化文本识别和停顿检测

## 使用场景

### 1. 实时语音助手

在 Unmute 等语音助手中，使用 2 秒窗口检测用户何时说完：

```python
PAUSE_PREDICTION_HEAD_INDEX = 2  # 2秒窗口
if pause_prediction > 0.5:
    # 用户已停止说话，可以开始响应
    trigger_assistant_response()
```

### 2. 转录应用

在转录应用中，可以使用不同的窗口来分段：

```python
# 使用较短的窗口（0.5秒）进行更细粒度的分段
if vad_heads[0][0, 0, 0] > 0.5:
    insert_pause_marker()
```

### 3. 可视化停顿

```bash
# 运行时显示 VAD 检测结果
uv run scripts/stt_from_mic_rust_server.py --show-vad
# 输出: Hello world | how are you | I'm fine
#              ↑ VAD 检测到的停顿
```

## 优势对比

### Semantic VAD vs 传统 VAD

| 特性 | Semantic VAD | 传统 VAD（能量检测） |
|------|--------------|---------------------|
| 检测依据 | 语义理解 + 音频特征 | 音频能量/频谱 |
| 误检率 | 低（理解语境） | 高（噪音敏感） |
| 延迟 | 与 ASR 相同 | 极低 |
| 计算开销 | 几乎无额外开销 | 很低 |
| 适用场景 | 对话系统、智能助手 | 预处理、降噪 |

### 关键优势

1. **语义感知**：能区分"嗯..."（思考停顿）和真正的结束
2. **集成高效**：无需额外模型，与 ASR 共享计算
3. **多时间窗口**：灵活适应不同应用场景
4. **低延迟**：实时输出，无需等待额外处理

## 技术细节

### 1. 模型配置

```toml
[modules.asr.model.extra_heads]
num_heads = 4    # 4个时间窗口
dim = 6          # 每个头的隐藏维度
```

### 2. 数据流

```
音频流 (24kHz)
  → 80ms 块 (1920 samples)
  → Mimi 编码 (12.5 fps)
  → Transformer 处理
  → 每帧输出:
      - 1 个文本 token
      - 4 个 VAD 概率值
```

### 3. 阈值选择

```python
# 标准阈值
if pause_prediction > 0.5:
    detect_pause()

# 可根据应用调整
AGGRESSIVE_THRESHOLD = 0.3  # 更快响应
CONSERVATIVE_THRESHOLD = 0.7  # 减少误检
```

## 代码示例

### 完整使用示例

```python
import moshi.models
import torch

# 加载模型（自动包含 VAD）
info = moshi.models.loaders.CheckpointInfo.from_hf_repo(
    "kyutai/stt-1b-en_fr"
)
lm = info.get_moshi(device="cuda", dtype=torch.bfloat16)
lm_gen = moshi.models.LMGen(lm, temp=0)

# 流式处理
with lm_gen.streaming(1):
    for audio_chunk in audio_stream:
        audio_tokens = mimi.encode(audio_chunk)

        # 获取文本和 VAD 预测
        text_tokens, vad_heads = lm_gen.step_with_extra_heads(audio_tokens)

        # 检查不同时间窗口
        pause_0_5s = vad_heads[0][0, 0, 0].item()
        pause_1_0s = vad_heads[1][0, 0, 0].item()
        pause_2_0s = vad_heads[2][0, 0, 0].item()
        pause_3_0s = vad_heads[3][0, 0, 0].item()

        # 根据应用需求选择合适的窗口
        if pause_2_0s > 0.5:
            print("User finished speaking!")
```

## 总结

Semantic VAD 是 Kyutai STT 的一个创新特性，通过在 ASR 模型中添加轻量级的预测头，实现了基于语义理解的停顿检测。它与 ASR 模型深度集成，共享特征提取，几乎不增加计算开销，同时提供了比传统 VAD 更准确、更智能的停顿检测能力。

**核心要点**：
- ✅ 集成在 ASR 模型中，不是独立模型
- ✅ 4 个预测头对应 4 个时间窗口（0.5s, 1s, 2s, 3s）
- ✅ 与文本预测共享 Transformer 特征
- ✅ 实时输出，无额外延迟
- ✅ 仅在 1B 模型中可用（2.6B 模型不包含）

## 参考资料

- [Kyutai STT 项目页面](https://kyutai.org/next/stt)
- [Semantic VAD 详细说明](https://kyutai.org/next/stt#semantic-vad)
- [Unmute 语音助手](https://github.com/kyutai-labs/unmute)
- [Delayed Streams Modeling 论文](https://arxiv.org/abs/2509.08753)
