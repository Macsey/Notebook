


## 1. 核心思想 

传统的 SGD 对所有参数 $\theta$ 使用相同的学习率 $\eta$。但在实际问题中（特别是稀疏数据，如 NLP 中的词向量），不同参数的更新频率往往不同：
- **频繁出现的特征**：更新次数多，希望学习率小一点。
- **稀疏的特征**：更新次数少，希望学习率大一点。
AdaGrad 的解决方式：
为每一个参数独立调整学习率。它通过记录过去所有梯度的平方和，来自动调节当前的学习率。
## 2. 算法公式 
假设：
- $\theta_t$：第 $t$ 步的参数。
- $g_t$：第 $t$ 步的梯度 ($g_t = \nabla_\theta J(\theta_t)$)。
- $\eta$：全局初始学习率。
- $\epsilon$：防止分母为 0 的极小值（通常取 $10^{-8}$）。
### 关键步骤：

1. 累积平方梯度 (Accumulate Squared Gradients)：
    维护一个变量 $s_t$ (有些教材用 $r_t$ 或 $h_t$)，它是从开始到现在所有梯度的平方累加和。
    $$s_t = s_{t-1} + g_t \odot g_t$$
    (注：$\odot$ 表示按元素相乘)
    
2. 参数更新 (Parameter Update)：
    
    计算更新量时，将全局学习率除以历史梯度平方和的平方根。
    $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t + \epsilon}} \odot g_t$$

## 3. 算法特点分析

### ✅ 优点 (Pros)

1. **自适应性强**：
    
    - **梯度大的参数**（历史累积 $s_t$ 大） $\rightarrow$ 分母大 $\rightarrow$ **学习率自动减小**。
        
    - **梯度小的参数**（历史累积 $s_t$ 小） $\rightarrow$ 分母小 $\rightarrow$ **学习率自动增大**。
        
2. **适合稀疏数据**：在处理稀疏特征（如推荐系统、文本分类）时表现优异，因为它能给予不常出现的特征更大的更新权重。
    
3. **对 $\eta$ 不敏感**：相比 SGD，AdaGrad 对初始学习率的敏感度较低。
    

### ❌ 缺点 (Cons)

1. **学习率单调递减（过早停止）**：
    
    - 公式中 $s_t$ 是**一直累加**正数（梯度平方）的。
        
    - 因此，$s_t$ 会越来越大，导致分母越来越大。
        
    - 最终，有效学习率 $\frac{\eta}{\sqrt{s_t + \epsilon}}$ 会趋近于 0，导致模型**提前停止学习**，无法收敛到最优解。
        
2. **依赖全局初始学习率**：虽然比 SGD 好，但仍需手动设置一个初始的 $\eta$。
    

---

## 4. 对比总结 (Summary Table)

|**特性**|**描述**|
|---|---|
|**全称**|Adaptive Gradient Algorithm|
|**核心机制**|累积历史梯度的平方和，用于缩放学习率|
|**适用场景**|处理**稀疏数据** (Sparse Data)|
|**主要问题**|训练后期学习率衰减过快 (Aggressive Decay)|
|**后继改进**|RMSProp, AdaDelta (旨在解决分母无限累加的问题)|

---

## 5. PyTorch 实现代码

在 PyTorch 中，使用非常简单，直接调用 `optim.Adagrad`：

Python

```
import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的模型
model = nn.Linear(10, 1)

# 定义优化器：AdaGrad
# lr: 初始学习率 (通常设为 0.01，比 SGD 的 0.001 要大一点)
# lr_decay: 学习率衰减 (可选)
# weight_decay: L2 正则化 (可选)
optimizer = optim.Adagrad(model.parameters(), lr=0.01)

# 模拟训练步骤
loss_fn = nn.MSELoss()
input_data = torch.randn(5, 10)
target = torch.randn(5, 1)

# 标准训练循环
optimizer.zero_grad()           # 清空梯度
output = model(input_data)      # 前向传播
loss = loss_fn(output, target)  # 计算损失
loss.backward()                 # 反向传播 (计算 g_t)
optimizer.step()                # 更新参数 (内部自动处理 s_t 和 theta)
```

---

下一步建议：

由于 AdaGrad 存在学习率衰减过快的问题，后续通常会学习 RMSProp 算法（它引入了衰减系数来解决这个问题）或 Adam 算法。

您需要我为您生成 **RMSProp** 的对比笔记，还是想看一看这几个算法在**收敛轨迹上的可视化对比图**？