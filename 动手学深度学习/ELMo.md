

### 1. 核心痛点：多义词的“精神分裂”

在 Word2Vec/GloVe 中：

- `bank` (银行) 的向量是 `[0.1, 0.5]`。
    
- `bank` (河岸) 的向量也是 `[0.1, 0.5]`。
    

**ELMo 的核心思想：**

词向量不应该是一成不变的查表结果，而应该是**通过一个深层网络实时算出来的函数结果**。

$$\text{Vector}_{\text{bank}} = f(\text{"bank"} \mid \text{Context})$$

---

### 2. 架构原理：双向 LSTM (Bi-LSTM)

这是 ELMo 和 BERT 最大的硬件区别：

- **BERT** 用的是 Transformer（并行，注意力机制）。
    
- **ELMo** 用的是 **LSTM**（串行，循环神经网络）。
    

ELMo 由两个独立的 LSTM 组成：

1. **前向 LSTM (Forward LSTM)**：
    
    - 从左往右读：`I` -> `ate` -> `an` -> `apple`。
        
    - 预测任务：根据上文预测下一个词。
        
2. **后向 LSTM (Backward LSTM)**：
    
    - 从右往左读：`apple` -> `an` -> `ate` -> `I`。
        
    - 预测任务：根据下文预测上一个词。
        

**关键点 (CS 考点)：**

ELMo 的双向是**“伪双向”**（Shallow Bidirectionality）。

- 它的前向和后向是**独立训练**的，最后只是简单地把两个向量**拼接 (Concatenate)** 起来。
    
- **BERT** 是**“真双向”**（Deep Bidirectionality），它的 Self-Attention 能同时看到左右两边，是深度融合。
    

---

### 3. ELMo 的绝活：层级加权 (Layer Aggregation)

BERT 通常取最后一层的输出，但 ELMo 认为**“每一层学到的东西都不一样”**：

- **底层 LSTM**：倾向于捕捉**语法**（Syntax）信息（比如词性：名词、动词）。
    
- **高层 LSTM**：倾向于捕捉**语义**（Semantics）信息（比如语境、多义词）。
    

**ELMo 的最终词向量，是所有层输出的加权和：**

$$\mathbf{ELMo}_w = \gamma \sum_{j=0}^{L} s_j \mathbf{h}_{w,j}$$

- $\mathbf{h}_{w,j}$：第 $j$ 层 LSTM 的输出向量。
    
- $s_j$：**可学习的权重 (Softmax-normalized weights)**。让下游任务自己决定它更需要语法（底层）还是语义（高层）。
    
- $\gamma$：缩放因子。
    

这意味着，做**词性标注 (POS Tagging)** 任务时，模型可能会自动给底层权重打高分；做**问答系统**时，模型会给高层权重打高分。

---

### 4. 训练方式：标准的语言模型 (Language Modeling)

和 BERT 的 MLM（完形填空）不同，ELMo 用的是最传统的**自回归语言模型**。

- **目标**：最大化似然概率 $P(w_1, w_2, ..., w_N)$。
    
- **做法**：
    
    - 给定 `The`，预测 `cat`。
        
    - 给定 `The cat`，预测 `sat`。
        
    - ...
        
- **缺点**：你看，预测 `cat` 时，它**只能看见左边**的 `The`，看不见右边的词。这就是为什么后来 BERT 要搞 MLM 的原因。
    

---

### 5. 使用方式：Feature-based (特征提取)

这是 ELMo 和 BERT 的另一个巨大分水岭。

- **BERT (Fine-tuning 模式)**：
    
    把 BERT 拿过来，**解冻所有参数**，在你的任务上微调整个网络。
    
- **ELMo (Feature-based 模式)**：
    
    把 ELMo 看作一个**冻结的特征提取器**。
    
    1. 输入句子。
        
    2. ELMo 算出每个词的向量。
        
    3. 把这些向量**喂给你自己的模型**（比如一个简单的 Linear 层或另一个 LSTM）。
        
    4. **训练时只更新你自己的模型参数，ELMo 的参数不动。**
        

---

### 6. ELMo vs BERT：为什么 BERT 赢了？

作为 CS 学生，你需要从计算特性上理解这次迭代：

|**特性**|**ELMo**|**BERT**|
|---|---|---|
|**基础单元**|LSTM (RNN)|Transformer Encoder|
|**计算并行性**|**差** (必须等上一个词算完才能算下一个)|**极好** (所有词同时计算)|
|**双向性质**|**拼接** (独立的前向+后向)|**深度融合** (Self-Attention 全局视野)|
|**训练目标**|预测下一个词 (Next Token Prediction)|完形填空 (Masked LM)|
|**使用方式**|特征提取 (Feature-based)|全参微调 (Fine-tuning)|
|**长距离依赖**|弱 (LSTM 梯度随距离衰减)|强 (Attention 直接连接任意两个词)|

### 总结

ELMo 是 **“动态词向量”的鼻祖**。

它告诉了全世界：**“别再查表了，词向量应该是算出来的，而且每一层都有用。”**

虽然它很快被 BERT 取代了，但它提出的 **“Contextualized Embeddings”** 概念，是现代 NLP 的基石。理解了 ELMo，你就理解了从 Word2Vec 到 BERT 进化的中间态。