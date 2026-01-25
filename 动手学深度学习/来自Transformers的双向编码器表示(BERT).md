[[ELMo]]
[[GPT]]

## 1. 背景

BERT 的出现是为了解决之前模型的两个核心痛点：**上下文缺失**和**单向限制**。

| **模型阶段**               | **代表模型**        | **特点**                     | **局限性**                                                   |
| ---------------------- | --------------- | -------------------------- | --------------------------------------------------------- |
| **1.0 静态词向量**          | Word2Vec, GloVe | 查表式，每个词向量固定                | **一词多义失效** (Context-Free)。如 "crane" (鹤/吊车) 向量完全一样。        |
| **2.0 动态上下文**          | ELMo            | 基于双向 LSTM                  | 上下文敏感，但**架构特定**，微调时需要针对任务修改架构。                            |
| **2.5 单向 Transformer** | GPT (早期)        | 基于 Transformer Decoder     | 上下文敏感，架构通用，但**只能从左向右看** (自回归)，无法利用下文信息。                   |
| **3.0 BERT**           | **BERT**        | **基于 Transformer Encoder** | **深度双向 (Deep Bidirectional)** + **任务无关 (Task-Agnostic)**。 |

**核心价值：** 开创了 **“预训练 (Pre-training) + 微调 (Fine-tuning)”** 的范式。预训练模型负责“读懂语言”，微调阶段只需加一个简单的输出层（Output Layer）即可适配 11 种不同的下游任务。


## 2. 输入表示：BERT 的“三合一”嵌入

BERT 的输入不仅仅是 Token ID，它是三个向量的**求和**。

$$Input = \text{TokenEmbed} + \text{SegmentEmbed} + \text{PositionalEmbed}$$

### A. 序列构造规则

- **特殊 Token**：
    
    - `<cls>`: 放在句首，用于存储**整个句子的分类特征** (Classification)。
        
    - `<sep>`: 分隔符 (Separator)。
        
- **格式**：
    
    - 单句：`<cls> Sentence <sep>`
        
    - 句对：`<cls> Sentence A <sep> Sentence B <sep>`
        

### B. 三种 Embedding 详解

1. **词元嵌入 (Token Embeddings)**:
    
    - 表示单词本身的语义。
        
    - 通常使用 **WordPiece** 分词。
        
2. **片段嵌入 (Segment Embeddings)**:
    
    - 用于区分句子 A 和句子 B。
        
    - 属于句子 A 的词全加向量 $e_A$，属于句子 B 的词全加向量 $e_B$。
        
3. **位置嵌入 (Position Embeddings)**:
    
    - **关键点**：与原始 Transformer (Sinusoidal) 不同，BERT 的位置向量是 **可学习的 (Learnable)**。
        
    - 最大长度通常限制为 1000 (或 512)。
        

---

## 3. 预训练任务 (Pre-training Tasks)

BERT 不依赖人工标注数据，而是通过两个**自监督任务**进行联合训练。

### 任务一：掩蔽语言模型 (Masked Language Model, MLM)

- **目标**：完形填空。利用双向上下文预测被盖住的词。
    
- **操作**：随机 Mask 掉 15% 的 Token。
    
- **80-10-10 策略** (为了解决预训练和微调时输入分布不一致的问题)：
    
    - **80%**：替换为 `<mask>` (让模型去猜)。
        
    - **10%**：替换为 **随机词** (强迫模型纠错，依赖上下文判断真伪)。
        
    - **10%**：保持 **原词不变** (告诉模型有时输入就是对的)。
        
- **代码实现**：
    
    - 输入：Transformer 最后一层的输出 `encoded_X`。
        
    - 结构：MLP (Linear -> ReLU -> LayerNorm -> Linear)。
        
    - 输出：词表大小的概率分布。
        

### 任务二：下一句预测 (Next Sentence Prediction, NSP)

- **目标**：二分类任务。判断句子 B 是否是 A 的真实下文。
    
- **数据构造**：
    
    - 50% 正样本 (IsNext)。
        
    - 50% 负样本 (NotNext，随机拼接)。
        
- **代码实现**：
    
    - 输入：**仅使用 `<cls>` 位置** 的向量。
        
    - 结构：Linear -> Tanh -> Linear (输出维度 2)。
        

---

## 4. 模型架构与代码结构 (CS 视角)

在 D2L 的 PyTorch 实现中，类结构如下：

1. **`BERTEncoder` (骨架)**
    
    - 本质就是标准的 **Transformer Encoder**。
        
    - 包含了 Embedding 层（Token+Segment+Pos）和 $N$ 层 `EncoderBlock`。
        
    - **参数**：`num_hiddens=768`, `num_layers=12`, `num_heads=12` (BERT-Base 配置)。
        
2. **`MaskLM` (MLM 头部)**
    
    - 负责处理 MLM 任务，只对被 Mask 的位置计算 Loss。
        
3. **`NextSentencePred` (NSP 头部)**
    
    - 负责处理 NSP 任务。
        
4. **`BERTModel` (整体封装)**
    
    - 前向传播逻辑：
        
        Python
        
        ```
        def forward(self, tokens, segments, ...):
            # 1. 骨干提取特征
            encoded_X = self.encoder(tokens, segments, ...)
        
            # 2. MLM 任务预测 (只预测被 Mask 的部分)
            mlm_hat = self.mlm(encoded_X, pred_positions)
        
            # 3. NSP 任务预测 (只用 CLS token)
            # 注意：CLS 通常会先过一个 MLP 层
            nsp_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        
            return encoded_X, mlm_hat, nsp_hat
        ```
        

---

## 5. 重点思考与练习

1. **为什么 BERT 收敛比 GPT 慢？**
    
    - GPT 每次预测下一个词，每个 token 都参与 Loss 计算。
        
    - BERT 每次只 Mask 15% 的词，只有这 15% 贡献 Loss，所以训练需要的 Step 更多。
        
2. **激活函数 GELU vs ReLU**：
    
    - BERT 原始论文使用的是 **GELU (Gaussian Error Linear Unit)**，它在 0 附近比 ReLU 更平滑，允许少量负数通过，有助于深层网络训练。
        
3. **梯度流向**：
    
    - `loss = loss_mlm + loss_nsp`。反向传播时，两个任务的梯度会混合，同时更新底层的 Encoder 参数。