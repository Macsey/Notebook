```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import math
import collections

# ==========================================
# 第一部分：配置与辅助函数
# ==========================================

# 设置随机种子，保证结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==========================================
# 第二部分：数据预处理 (Data Processing)
# ==========================================

class PTBDataProcess:
    """
    负责将原始文本转换为训练所需的索引列表，包含：
    1. 建立词表 (Vocabulary)
    2. 将文本转为 ID
    3. 下采样 (Subsampling) 去除过高频的词 (如 "the", "a")
    """
    def __init__(self, sentences, min_freq=1):
        self.sentences = sentences # 原始句子列表
        self.min_freq = min_freq   # 最小词频，低于此频率的词会被丢弃
        
        # 1. 统计词频
        # counter: {word: count}
        self.counter = collections.Counter([tk for line in sentences for tk in line])
        
        # 2. 构建映射关系
        # idx_to_token: [word0, word1, ...]
        # token_to_idx: {word0: 0, word1: 1, ...}
        self.idx_to_token = ['<unk>']
        self.token_to_idx = {'<unk>': 0}
        
        # 按词频从高到低排序，构建词表
        for token, freq in sorted(self.counter.items(), key=lambda x: x[1], reverse=True):
            if freq >= min_freq:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
                
        # 3. 将文本转换为 ID 序列
        self.corpus = []
        for line in sentences:
            # 如果词不在词表中，转为 <unk> (ID=0)
            self.corpus.append([self.token_to_idx.get(tk, 0) for tk in line])
            
        # 4. 执行下采样
        self.subsampled_corpus = self._subsample()

    def _subsample(self):
        """
        下采样逻辑：
        对于高频词 w，我们有 P(drop) 的概率丢弃它。
        公式：P(keep) = sqrt(threshold / freq)
        """
        subsampled = []
        # 计算总词数
        total_tokens = sum(len(line) for line in self.corpus)
        threshold = 1e-4 # 阈值，越小丢弃的高频词越多
        
        for line in self.corpus:
            sub_line = []
            for token_id in line:
                token_str = self.idx_to_token[token_id]
                count = self.counter[token_str]
                freq = count / total_tokens
                
                # 计算保留概率
                keep_prob = math.sqrt(threshold / freq)
                
                # 随机决定是否保留
                if random.uniform(0, 1) < keep_prob:
                    sub_line.append(token_id)
            subsampled.append(sub_line)
        return subsampled

# ==========================================
# 第三部分：数据集与负采样 (Dataset & Negative Sampling)
# ==========================================

class Word2VecDataset(Dataset):
    """
    PyTorch 数据集：负责产生 (中心词, 上下文+负样本) 的训练对
    """
    def __init__(self, processed_data, window_size=2, num_negatives=5):
        self.data = processed_data
        self.window_size = window_size
        self.num_negatives = num_negatives
        
        # 1. 获取所有的中心词和正样本上下文
        self.centers, self.contexts = self._get_centers_and_contexts()
        
        # 2. 预先计算采样权重 (用于负采样)
        # 论文建议：P(w) ~ count(w)^0.75
        vocab_size = len(self.data.idx_to_token)
        counts = [self.data.counter.get(self.data.idx_to_token[i], 0) for i in range(vocab_size)]
        # 注意：为了防止对 <unk> 采样过多，可以特殊处理，这里简单处理
        self.sampling_weights = [c**0.75 for c in counts]
        
        # 准备好所有词的索引列表，用于 random.choices
        self.population = list(range(vocab_size))

    def _get_centers_and_contexts(self):
        """滑动窗口提取中心词和上下文"""
        centers, contexts = [], []
        for line in self.data.subsampled_corpus:
            if len(line) < 2: continue # 句子太短跳过
            
            for i in range(len(line)):
                # 动态窗口大小：1 到 window_size 随机，增强鲁棒性
                # actual_window = random.randint(1, self.window_size)
                actual_window = self.window_size # 这里为了演示稳定，固定窗口
                
                start = max(0, i - actual_window)
                end = min(len(line), i + actual_window + 1)
                
                # 上下文索引 (排除中心词自己)
                ctx_indices = [line[j] for j in range(start, end) if j != i]
                
                if ctx_indices:
                    centers.append(line[i])
                    contexts.append(ctx_indices)
        return centers, contexts

    def __getitem__(self, idx):
        """
        获取一个样本：
        返回：(中心词, 正样本上下文列表, 负样本列表)
        """
        center = self.centers[idx]
        context = self.contexts[idx]
        
        # 负采样逻辑：
        # 需要采样的数量 = 当前上下文数量 * num_negatives
        k = len(context) * self.num_negatives
        
        # 使用加权随机采样
        negatives = random.choices(
            self.population, 
            weights=self.sampling_weights, 
            k=k
        )
        
        return center, context, negatives

    def __len__(self):
        return len(self.centers)

def batchify(data):
    """
    Collate Function: 将 batch 内不同长度的数据进行 Padding 对齐
    """
    # data 是一个 list，包含 batch_size 个 (center, context, negative) 元组
    
    # 1. 计算当前 batch 中最长的序列长度 (context + negative)
    max_len = max(len(c) + len(n) for _, c, n in data)
    
    centers = []
    contexts_negatives = []
    masks = []
    labels = []
    
    for center, context, negative in data:
        centers.append([center])
        
        # 拼接正负样本：[正, 正, ..., 负, 负, ...]
        cur_ctx_neg = context + negative
        
        # 生成标签：正样本为 1，负样本为 0
        cur_labels = [1] * len(context) + [0] * len(negative)
        
        # 生成掩码 (Mask)：有效位置为 1，填充位置为 0
        cur_mask = [1] * len(cur_ctx_neg)
        
        # Padding: 补 0 直到 max_len
        padding_len = max_len - len(cur_ctx_neg)
        cur_ctx_neg += [0] * padding_len
        cur_labels += [0] * padding_len
        cur_mask += [0] * padding_len
        
        contexts_negatives.append(cur_ctx_neg)
        labels.append(cur_labels)
        masks.append(cur_mask)
    
    # 转为 Tensor
    # centers: (batch_size, 1)
    # contexts_negatives: (batch_size, max_len)
    # labels: (batch_size, max_len)
    # masks: (batch_size, max_len)
    return (torch.tensor(centers), 
            torch.tensor(contexts_negatives), 
            torch.tensor(labels).float(), 
            torch.tensor(masks).float())

# ==========================================
# 第四部分：模型定义 (Skip-Gram Model)
# ==========================================

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipGramModel, self).__init__()
        
        # 中心词矩阵 V (Input Vectors)
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # 上下文矩阵 U (Output Vectors)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # 初始化权重 (小的随机数)
        init_range = 0.5 / embed_dim
        self.center_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)

    def forward(self, center_ids, target_ids):
        """
        center_ids: (batch_size, 1)
        target_ids: (batch_size, max_len) -> 包含正样本和负样本
        """
        # 1. 查表
        # v_c: (batch_size, 1, dim)
        v_c = self.center_embeddings(center_ids)
        
        # u_t: (batch_size, max_len, dim)
        u_t = self.context_embeddings(target_ids)
        
        # 2. 计算点积 (Dot Product)
        # 我们需要计算 v_c 和每一个 u_t 的相似度
        # bmm (Batch Matrix Multiplication): (b, max_len, d) * (b, d, 1) -> (b, max_len, 1)
        # v_c.permute(0, 2, 1) 将形状变为 (b, dim, 1) 以便矩阵乘法
        scores = torch.bmm(u_t, v_c.permute(0, 2, 1))
        
        # 3. 去掉多余的维度
        # (b, max_len, 1) -> (b, max_len)
        return scores.squeeze(2)

# ==========================================
# 第五部分：验证工具 (Evaluation)
# ==========================================

def get_similar_words(word, model, token_to_idx, idx_to_token, k=5):
    """
    计算余弦相似度，寻找最近邻
    """
    if word not in token_to_idx:
        print(f"Word '{word}' not in vocabulary.")
        return
    
    # 切换到评估模式
    model.eval()
    
    # 获取查询词向量 (1, dim)
    word_id = torch.tensor([token_to_idx[word]])
    v_w = model.center_embeddings(word_id)
    
    # 获取所有词向量矩阵 (V, dim)
    all_embs = model.center_embeddings.weight
    
    # 归一化 (Normalization) -> 方便计算 Cosine Similarity
    # dim=1 表示沿向量维度计算范数
    v_w = v_w / v_w.norm(dim=1, keepdim=True)
    all_embs = all_embs / all_embs.norm(dim=1, keepdim=True)
    
    # 计算相似度: (1, dim) * (dim, V) -> (1, V)
    # .t() 是转置
    similarity = torch.mm(v_w, all_embs.t()).squeeze()
    
    # 取出 Top K (包含它自己，所以取 k+1)
    values, indices = torch.topk(similarity, k + 1)
    
    print(f"\nTarget word: {word}")
    for i in range(1, k + 1):
        idx = indices[i].item()
        sim = values[i].item()
        print(f"  - {idx_to_token[idx]}: {sim:.4f}")

# ==========================================
# 第六部分：主程序 (Main Execution)
# ==========================================

if __name__ == "__main__":
    # --- 1. 准备简单语料 ---
    # 为了演示效果，我们手动构造一些有明显语义关系的句子
    # 让模型学习 "king"-"queen", "man"-"woman", "apple"-"fruit" 的关系
    raw_text = [
        "the king loves the queen",
        "the queen loves the king",
        "the king is a man",
        "the queen is a woman",
        "man and woman are humans",
        "apple is a fruit",
        "banana is a fruit",
        "orange is a fruit",
        "i eat apple",
        "i eat banana",
        "dog is an animal",
        "cat is an animal"
    ]
    # 简单的分词
    sentences = [s.split() for s in raw_text] * 100 # 复制多次以增加训练步数

    # --- 2. 数据处理 ---
    print("Processing data...")
    processor = PTBDataProcess(sentences, min_freq=1)
    print(f"Vocabulary Size: {len(processor.idx_to_token)}")
    
    dataset = Word2VecDataset(processor, window_size=2, num_negatives=5)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=batchify)

    # --- 3. 初始化模型 ---
    EMBED_DIM = 10 # 词向量维度 (Toy data用小维度即可)
    VOCAB_SIZE = len(processor.idx_to_token)
    model = SkipGramModel(VOCAB_SIZE, EMBED_DIM)

    # --- 4. 训练配置 ---
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # BCEWithLogitsLoss = Sigmoid + BCELoss
    # reduction='none' 很关键：因为我们要自己处理 Mask，不能让它自动求平均
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # --- 5. 训练循环 ---
    print("\nStart Training...")
    EPOCHS = 50
    
    for epoch in range(EPOCHS):
        total_loss = 0
        model.train() # 切换到训练模式
        
        for step, (centers, contexts_negatives, labels, masks) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward: (batch_size, max_len)
            pred_scores = model(centers, contexts_negatives)
            
            # Loss: (batch_size, max_len)
            loss_matrix = criterion(pred_scores, labels)
            
            # Masking: 只计算有效数据的 loss，填充部分乘以 0
            masked_loss = loss_matrix * masks
            
            # Mean Loss: 总 Loss / 有效样本数 (Mask 之和)
            loss = masked_loss.sum() / masks.sum()
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

    # --- 6. 验证结果 ---
    print("\nTraining Finished. Checking Results...")
    # 看看 King 和 Queen 是否接近
    get_similar_words("king", model, processor.token_to_idx, processor.idx_to_token)
    # 看看 Apple 和 Fruit 是否接近
    get_similar_words("apple", model, processor.token_to_idx, processor.idx_to_token)
```