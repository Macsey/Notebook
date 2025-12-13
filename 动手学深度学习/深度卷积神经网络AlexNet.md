## 一、AlexNet 背景与核心意义

### 1. 历史地位

- 2012 年 ImageNet 挑战赛冠军模型，首次证明**端到端深度学习**在计算机视觉任务上超越手工设计特征（如 SIFT、HOG），是深度学习在 CV 领域爆发的标志性模型。

- 作者：Alex Krizhevsky、Ilya Sutskever、Geoff Hinton，以第一作者命名为 AlexNet。

### 2. 突破的关键前提

|   |   |
|---|---|
|关键要素|具体说明|
|数据基础|ImageNet 数据集（1000 类、120 万张图像），解决了早期小数据集无法支撑深层模型训练的问题|
|硬件支撑|首次基于 GPU（2 块 NVIDIA GTX 580，3GB 显存）并行训练，突破 CPU 算力瓶颈（卷积、矩阵乘法可高度并行）|
|技术积累|解决了深层模型训练的核心难题：ReLU 激活函数（替代 sigmoid）、Dropout 正则化、参数初始化、数据增强|

## 二、AlexNet 核心架构（PyTorch 实现）

### 1. 架构对比：AlexNet vs LeNet

|       |                   |                             |
| ----- | ----------------- | --------------------------- |
| 对比维度  | LeNet-5           | AlexNet                     |
| 网络深度  | 5 层（2 卷积 + 3 全连接） | 8 层（5 卷积 + 3 全连接）           |
| 卷积通道数 | 最多 16 通道          | 最多 384 通道（是 LeNet 的 10 倍以上） |
| 激活函数  | sigmoid           | ReLU（解决梯度消失，计算更快）           |
| 正则化   | 仅权重衰减             | Dropout（全连接层）+ 数据增强         |
| 输入尺寸  | 32×32（MNIST）      | 224×224（ImageNet）           |

### 2. PyTorch 完整实现代码

```python
import torch

from torch import nn

from d2l import torch as d2l

# 定义AlexNet网络结构

net = nn.Sequential(

# 第1卷积块：11×11卷积（捕捉大目标）→ ReLU → 3×3最大池化（步幅2，压缩尺寸）

nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),

nn.MaxPool2d(kernel_size=3, stride=2),

# 第2卷积块：5×5卷积（减小窗口）→ ReLU → 3×3最大池化

nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),

nn.MaxPool2d(kernel_size=3, stride=2),

# 第3-5卷积块：3×3卷积（精细特征）→ ReLU（无池化，保留尺寸）

nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),

nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),

nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),

nn.MaxPool2d(kernel_size=3, stride=2), # 最后一次池化

# 全连接层：展平 → 4096维 → ReLU → Dropout（0.5）

nn.Flatten(),

nn.Linear(6400, 4096), nn.ReLU(), # 6400=256×5×5（池化后尺寸）

nn.Dropout(p=0.5),

# 第2全连接层：4096维 → ReLU → Dropout

nn.Linear(4096, 4096), nn.ReLU(),

nn.Dropout(p=0.5),

# 输出层：Fashion-MNIST为10类（原论文ImageNet为1000类）

nn.Linear(4096, 10)

)
```

### 3. 各层输出形状验证（输入：1×1×224×224，单样本单通道）

```python
X = torch.randn(1, 1, 224, 224) # 模拟输入（batch_size=1, channel=1, H=224, W=224）

for layer in net:

X = layer(X)

print(f"{layer.__class__.__name__:10s} output shape: {X.shape}")
```

**输出结果**（对应架构逻辑，验证尺寸正确性）：

```
Conv2d output shape: torch.Size([1, 96, 54, 54]) # 224→(224-11+2×1)/4 +1=54

ReLU output shape: torch.Size([1, 96, 54, 54])

MaxPool2d output shape: torch.Size([1, 96, 26, 26]) # 54→(54-3)/2 +1=26

Conv2d output shape: torch.Size([1, 256, 26, 26]) # (26-5+2×2)/1 +1=26

ReLU output shape: torch.Size([1, 256, 26, 26])

MaxPool2d output shape: torch.Size([1, 256, 12, 12]) # 26→(26-3)/2 +1=12

Conv2d output shape: torch.Size([1, 384, 12, 12]) # (12-3+2×1)/1 +1=12

ReLU output shape: torch.Size([1, 384, 12, 12])

Conv2d output shape: torch.Size([1, 384, 12, 12])

ReLU output shape: torch.Size([1, 384, 12, 12])

Conv2d output shape: torch.Size([1, 256, 12, 12])

ReLU output shape: torch.Size([1, 256, 12, 12])

MaxPool2d output shape: torch.Size([1, 256, 5, 5]) # 12→(12-3)/2 +1=5

Flatten output shape: torch.Size([1, 6400]) # 256×5×5=6400

Linear output shape: torch.Size([1, 4096])

ReLU output shape: torch.Size([1, 4096])

Dropout output shape: torch.Size([1, 4096])

Linear output shape: torch.Size([1, 4096])

ReLU output shape: torch.Size([1, 4096])

Dropout output shape: torch.Size([1, 4096])

Linear output shape: torch.Size([1, 10])
```

## 三、数据集处理（Fashion-MNIST 适配）

### 1. 问题与解决方案

- 原 AlexNet 输入为 224×224，而 Fashion-MNIST 为 28×28，需通过resize=224放大图像（虽非最优，但适配架构）。

- 数据加载代码（批量大小 128，含归一化、随机打乱）：

```python
batch_size = 128

# 加载训练集和测试集，resize到224×224

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
```

## 四、模型训练（PyTorch）

### 1. 训练参数设置

- 学习率：lr=0.01（比 LeNet 小，因 AlexNet 更深更广，需缓慢更新参数）

- 迭代轮数：num_epochs=10（Fashion-MNIST 数据量较小，10 轮足够收敛）

- 设备：优先使用 GPU（d2l.try_gpu()自动检测）

### 2. 训练代码（调用 d2l 工具函数，含损失计算、精度统计、可视化）

```python
lr, num_epochs = 0.01, 10

# 调用d2l.train_ch6训练，返回训练损失、精度及测试精度

d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

### 3. 典型训练结果

- 最终指标：训练损失≈0.33，训练精度≈88%，测试精度≈89%

- 训练速度：约 3900 样本 / 秒（GPU：NVIDIA Tesla V100 级别）

- 关键结论：AlexNet 在 Fashion-MNIST 上表现优于 LeNet（LeNet 测试精度≈84%），验证了深层架构的优势。

## 五、核心技术亮点与总结

### 1. 关键创新点（奠定现代 CNN 基础）

1. **ReLU 激活函数**：替代 sigmoid，解决梯度消失问题，计算效率提升 10 倍以上。

2. **Dropout 正则化**：全连接层使用 0.5 dropout，有效抑制过拟合（LeNet 仅用权重衰减）。

3. **数据增强**：训练时随机翻转、裁切、变色，扩充数据多样性，提升泛化能力。

4. **大卷积核 + 多通道**：11×11 大核捕捉全局特征，通道数从 96→384，提升特征表达能力。

5. **GPU 并行训练**：首次实现深层 CNN 的 GPU 加速，为后续大模型训练提供范式。

### 2. 局限性（现代视角）

- 全连接层参数过多（4096×4096×2≈3300 万参数），占总参数 80% 以上，显存占用大。

- 未使用批量归一化（BN），训练稳定性依赖学习率调整（BN 在 2015 年提出）。

- 对小尺寸图像（如 28×28）适配性差，需放大图像（丢失细节）。

