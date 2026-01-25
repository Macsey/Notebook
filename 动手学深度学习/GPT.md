讲完 BERT，接下来的主角毫无疑问就是 **GPT**。

如果说 BERT 是 NLP 里的“阅读理解专家”（擅长做选择题、填空题），那么 GPT 就是 **“创意写作专家”**（擅长写作文、写代码、聊天）。

GPT 全称 **G**enerative **P**re-trained **T**ransformer。虽然名字听起来和 BERT 像亲兄弟，但它们的**基因（架构）**和**成长路径（训练目标）**截然不同。

我们从 **架构原理、训练方式、以及它如何进化到 ChatGPT** 三个维度来拆解。



### 1. 架构原理：Transformer 的“另一半”

你还记得 Transformer 由 Encoder（编码器）和 Decoder（解码器）组成吗？

- **BERT** 拿走了 Encoder（双向，看全局）。
    
- **GPT** 拿走了 **Decoder**（单向，只看前面）。
    

#### 核心区别：掩蔽自注意力 (Masked Self-Attention)

这是 GPT 的灵魂。

在 BERT 的 Self-Attention 中，"love" 可以看到后面的 "coding"。

但在 GPT 中，**这是绝对禁止的**。

- **规则：** 当我生成第 $t$ 个词时，我只能看 $1$ 到 $t-1$ 的词。绝对不能偷看第 $t+1$ 个词。
    
- **实现：** 在 Attention 矩阵上加一个**上三角掩码 (Upper Triangular Mask)**。把对角线右上角全部设为 $-\infty$（经过 Softmax 后变成 0）。
    

**为什么？**

因为 GPT 的目标是**生成**。就像你说话一样，你说出的下一个字，只能基于你之前说过的话，而不能基于你还没说出来的话。


### 2. 训练目标：标准的语言模型 (Causal LM)

BERT 做的是“完形填空”（猜中间），GPT 做的是 **“接龙”**（猜后面）。

- **数学定义：** 最大化给定前文预测下一个词的概率。
    
    $$L = \sum_t \log P(u_t | u_{t-k}, \dots, u_{t-1}; \Theta)$$
    
- **直观理解：**
    
    - Input: "The quick brown fox"
        
    - Target: "jumps"
        
    - Input: "The quick brown fox jumps"
        
    - Target: "over"
        

**CS 视角对比：**

- **BERT:** 每次训练能同时学到一句话里 15% 的 Mask。效率低，但学得深。
    
- **GPT:** 每次训练都在预测**每一个** Token 的下一个词。效率高，但只能利用单向信息。
    

---

### 3. GPT 家族的进化史：从“平庸”到“神”

作为 CS 学生，你需要理解 GPT 是如何通过 **“Scaling Law（缩放定律）”** 产生质变的。

#### **GPT-1 (2018): "我也是个微调模型"**

- **参数量：** 1.17 亿 (和 BERT-Base 差不多)。
    
- **定位：** 和 BERT 一样，先预训练，然后针对下游任务（分类、蕴含）进行 **Fine-tuning**。
    
- **结局：** 效果不如 BERT。因为单向模型做分类任务确实不如双向模型（BERT）好用。
    

#### **GPT-2 (2019): "我不想微调了" (Zero-shot)**

- **参数量：** 15 亿。
    
- **核心洞察：** OpenAI 发现，只要模型够大、数据够多，它不需要微调就能做任务。
    
    - 你给它输入："English: Hello, French: "
        
    - 它自动补全："Bonjour"
        
- **结论：** 这是一个通用的多任务学习器。
    

#### **GPT-3 (2020): "大力出奇迹" (Few-shot / In-context Learning)**

- **参数量：** **1750 亿** (恐怖的 100 倍增长)。
    
- **核心创新：** **Prompt Engineering (提示工程)** 的诞生。
    
    - 你不需要更新模型参数（不微调）。
        
    - 你只需要在 Prompt 里给它几个例子（Few-shot），它就能瞬间学会新任务。
        
    - 比如给几个 SQL 的例子，它就能写 SQL。
        

#### **InstructGPT / ChatGPT (2022): "听得懂人话" (Alignment)**

- **痛点：** GPT-3 虽然博学，但像个“杠精”或“复读机”，不一定按人类指令行事。
    
- **解决方案：** **RLHF (Reinforcement Learning from Human Feedback)**。
    
    1. 找人写标准答案，微调 GPT (SFT)。
        
    2. 找人给 GPT 的回答打分，训练一个 Reward Model。
        
    3. 用强化学习（PPO 算法）让 GPT 即使在没人教的情况下，也能生成让人类满意的回答。
        

---

### 4. 代码实现：自回归生成循环 (Auto-regressive Loop)

BERT 的推理是一次性的（Forward 一次出结果），但 GPT 的推理是一个 **For 循环**。

这在工程上非常重要：

Python

```
import torch

def generate(model, input_ids, max_length=20):
    # input_ids: [Batch, Seq_Len] (比如 "Thinking is")
    
    for _ in range(max_length):
        # 1. 前向传播
        outputs = model(input_ids)
        
        # 2. 获取最后一个 Token 的 Logits
        # GPT 预测的是序列的下一个词
        next_token_logits = outputs.logits[:, -1, :] 
        
        # 3. 贪婪采样 (选概率最大的)
        # 实际中通常用 Top-k 或 Top-p 采样
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        
        # 4. 【关键】把新生成的词拼接到输入后面
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # 5. 循环，把 "Thinking is hard" 再次喂给模型...
        
    return input_ids
```

### 5. BERT vs. GPT：终极对比

|**维度**|**BERT**|**GPT**|
|---|---|---|
|**全称**|Bidirectional Encoder...|Generative Pre-trained Transformer|
|**架构组件**|Transformer **Encoder**|Transformer **Decoder**|
|**视野**|**双向** (能看到后文)|**单向** (只能看前文)|
|**注意力机制**|全局 Self-Attention|**Masked** Self-Attention|
|**训练目标**|完形填空 (MLM)|预测下一个词 (Next Token Prediction)|
|**擅长领域**|理解 (分类、实体识别、问答)|**生成** (写作、对话、代码补全)|
|**使用方式**|主要是微调 (Fine-tuning)|主要是提示 (Prompting)|

### 总结

GPT 就是一个 **“看过整个互联网数据的、带掩码的 Transformer Decoder”**。

- 它通过 **“预测下一个词”** 这一简单的目标，学会了语法、逻辑甚至世界知识。
    
- 它的成功证明了：只要模型够大（Scale），量变会引起质变（Emergent Abilities，涌现能力）。
    

接下来在 D2L 或深度学习的学习中，你会发现现在的趋势是 **Decoder-only** 架构（像 LLaMA, DeepSeek）正在统治世界，因为生成式任务（AIGC）的价值现在远超单纯的分类任务。