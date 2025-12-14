
## 1. 核心思想 

**为了解决 AdaGrad “学习率过早消失” 的问题。**

- **AdaGrad 的痛点**：它累积了**从开始到现在所有**梯度的平方和。随着训练进行，累积值 $s_t$ 无限增大，导致学习率分母无限变大，最终步长趋近于 0，模型在还没收敛时就“不动了”。
- **RMSProp 的改进**：它不再累积所有历史，而是使用**指数加权移动平均**
    - 通俗理解：它只关注**最近一段时间**的梯度大小，让“旧”的梯度记录慢慢被遗忘。
    - 这使得算法在非凸优化问题中，既能自适应调整学习率，又能保持持续的学习能力。
## 2. 算法公式 (Mathematical Formulation)
假设：
- $g_t$：第 $t$ 步的梯度。
- $s_t$：梯度平方的指数加权移动平均值。
- $\eta$：全局初始学习率。
    
- $\beta$ (或 $\alpha$)：**衰减率 (Decay Rate)**，控制历史信息的遗忘程度。通常设为 **0.9**。
    
- $\epsilon$：防止分母为 0 的极小值（通常取 $10^{-8}$）。
    

### 关键步骤：

1. 计算梯度平方的移动平均 (Moving Average of Squared Gradients)：
    
    不再直接相加，而是加权混合“旧状态”和“新梯度”。
    
    $$s_t = \beta \cdot s_{t-1} + (1 - \beta) \cdot (g_t \odot g_t)$$
    
    - _如果 $\beta=0.9$，意味着 $s_t$ 包含了约 90% 的历史信息和 10% 的当前梯度信息。_
        
2. 参数更新 (Parameter Update)：
    
    和 AdaGrad 形式一样，只是 $s_t$ 的计算方式变了。
    
    $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t + \epsilon}} \odot g_t$$
    

---

## 3. 算法特点分析

### ✅ 优点 (Pros)

1. **解决学习率急剧下降问题**：由于 $s_t$ 是移动平均，它不会像 AdaGrad 那样无限制单调递增。分母保持在一个合理的范围内，使得训练可以持续进行。
    
2. **适应非平稳目标**：在 Loss landscape（损失曲面）变化复杂的场景（如 RNN）中表现很好，因为它能快速适应当前的地形变化。
    
3. **自适应能力**：保留了 AdaGrad 的优点，对梯度大的参数减小步长，对梯度小的参数增大步长。
    

### ❌ 缺点 (Cons)

1. **仍需设置全局学习率**：虽然分母是自适应的，但分子上的 $\eta$ 仍然需要人工调参。
    
2. **可能陷入局部极小值**：在某些特定情况下，RMSProp 可能因为过度调节步长而在局部震荡（但在神经网络中通常不是大问题）。
    

---

## 4. 对比总结 (Summary Table)

|**特性**|**描述**|
|---|---|
|**提出者**|Geoffrey Hinton (Coursera Lecture 6e)|
|**核心机制**|使用**指数加权移动平均**来计算梯度的二阶矩（平方）|
|**超参数 $\beta$**|通常设为 **0.9** (PyTorch 中参数名为 `alpha`)|
|**适用场景**|**RNN (循环神经网络)**、非平稳环境、一般深度学习任务|
|**与 AdaGrad 区别**|AdaGrad 累积**所有**历史 (SUM) $\rightarrow$ 学习率必减<br><br>  <br><br>RMSProp 关注**最近**历史 (EMA) $\rightarrow$ 学习率可升可降|

---

## 5. PyTorch 实现代码

在 PyTorch 中，对应 `optim.RMSprop`。

Python

```
import torch
import torch.nn as nn
import torch.optim as optim

# 定义模型
model = nn.Linear(10, 1)

# 定义优化器：RMSProp
# lr: 学习率 (通常设为 0.001 或 0.01)
# alpha: 平滑常数 (对应公式中的 beta，默认 0.99，Hinton 课件建议 0.9)
# eps: 分母稳定项 (默认 1e-8)
# weight_decay: L2 正则化
# momentum: RMSProp 也可以结合动量使用 (通常设为 0)
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

# 模拟训练步骤
loss_fn = nn.MSELoss()
input_data = torch.randn(5, 10)
target = torch.randn(5, 1)

optimizer.zero_grad()
output = model(input_data)
loss = loss_fn(output, target)
loss.backward()
optimizer.step()
```

---

下一步建议：

我们已经复习了 SGD $\rightarrow$ AdaGrad $\rightarrow$ RMSProp。

现在的“版本之子” Adam (Adaptive Moment Estimation) 算法，实际上就是 Momentum (动量法) 和 RMSProp 的结合体。

您准备好把这两块拼图拼起来，查看 **Adam 算法** 的笔记了吗？或者您需要对比一下 Momentum 和 RMSProp 的区别？