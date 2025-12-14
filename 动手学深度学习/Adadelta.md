
- **核心目的**：
    
    1. 解决 Adagrad 学习率急剧下降的问题（这一点与 RMSProp 相同）。
    2. **解决传统优化算法中单位（量纲）不一致的问题**，并试图**消除对全局学习率超参数的依赖**
- **一句话概括**：Adadelta 像是 RMSProp 的升级版，但它不需要设置初始学习率，而是用过去的参数更新量来自动计算步长。
 

## Adadelta 试图解决两个主要问题：

1. **学习率单调递减 (The Decaying Learning Rate Problem)**：
    
    - Adagrad 累积所有历史梯度平方，导致分母过大，训练后期无法学习。
        
    - _解决方案_：使用指数加权移动平均（同 RMSProp）。
        
2. **单位不一致性 (Unit Mismatch)**：
    - 在 SGD 或 Adagrad 中，参数更新公式通常是 $\theta_{new} = \theta - \eta \cdot g$。
    - 如果参数 $\theta$ 的单位是米($m$)，梯度的单位通常是 $1/m$，那么更新量的单位就变成了 $1/m$（假设学习率无单位）。这在物理上是不合理的（你应该用米去更新米，而不是用 1/米）。
    - _解决方案_：Adadelta 引入了二阶牛顿法的思想，通过维护“参数更新量的移动平均”来修正单位，使其具有正确的量纲。


## 算法原理
Adadelta 维护了两个状态变量：

1. **梯度平方的移动平均** $E[g^2]_t$
2. **参数更新量平方的移动平均** $E[\Delta \theta^2]_t$

### 核心步骤 

1. **计算梯度**：$g_t$
2. 累积梯度平方 (同 RMSProp)：$$E[g^2]_t = \rho E[g^2]_{t-1} + (1 - \rho) g_t^2$$
    - $\rho$：衰减系数 (Decay Rate)，类似 RMSProp 的 $\beta$。
    - $RMS[g]_t = \sqrt{E[g^2]_t + \epsilon}$
3. 计算更新步长 (Adadelta 的精髓)：
    此处不使用全局学习率 $\eta$，而是使用上一时刻的参数更新量的 RMS 来近似：
    $$\Delta \theta_t = - \frac{RMS[\Delta \theta]_{t-1}}{RMS[g]_t} \cdot g_t$$
    即：$$\Delta \theta_t = - \frac{\sqrt{E[\Delta \theta^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t$$
    
4. 累积参数更新量平方 (为下一步做准备)：$$E[\Delta \theta^2]_t = \rho E[\Delta \theta^2]_{t-1} + (1 - \rho) \Delta \theta_t^2$$
5. 更新参数：$$\theta_{t+1} = \theta_t + \Delta \theta_t$$
##   关键特点
1. 无需设置默认学习率：
    这是 Adadelta 最显著的特征。公式中没有 $\eta$，它完全由算法内部的状态动态决定。
2. 单位一致：
    通过分子中的 $\sqrt{E[\Delta \theta^2]}$，更新量的单位被修正回了参数本身的单位。
3. 计算代价略高：
    相比 RMSProp，Adadelta 需要额外存储一个变量 $E[\Delta \theta^2]$，显存占用稍多一点。

## Adadelta vs RMSProp

|**特性**|**RMSProp**|**Adadelta**|
|---|---|---|
|**梯度累积**|指数加权移动平均|指数加权移动平均|
|**学习率**|**需要手动设置** (如 0.001)|**不需要设置** (自适应)|
|**状态变量**|1个 ($s_t$)|2个 ($E[g^2]$ 和 $E[\Delta \theta^2]$)|
|**主要公式**|$\Delta \theta = -\frac{\eta}{\sqrt{s_t}} g_t$|$\Delta \theta = -\frac{\sqrt{E[\Delta \theta^2]}}{\sqrt{E[g^2]}} g_t$|

##  总结 

- **Adadelta** 是对 Adagrad 的改进，旨在解决学习率衰减过快的问题。
- 它与 **RMSProp** 非常相似（都用了指数移动平均），但 Adadelta 更进一步，移除了全局学习率超参数。
- **适用场景**：如果你不想费心去调试学习率，或者你的模型对学习率非常敏感，Adadelta 是一个不错的备选方案。
- **现状**：虽然设计很精妙，但在目前的深度学习实践中，**Adam** 和 **RMSProp** 的使用率通常高于 Adadelta。

