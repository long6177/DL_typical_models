import torch
from torch import nn
import math

def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):
    """Scaled Dot-product Attention"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # query的形状: (batch_size, 查询个数, d)
    # keys的形状: (batch_size, "键值对"个数, d)
    # values的形状: (batch_size, "键值对"个数, 值的维度)
    # valid_lens的形状: (batch_size, ) 或者(batch_size, 查询个数)

    def forward(self, queries, keys, values, valid_lens=None):
        d  = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度(即batch_size里面一个进行转置)
        scores = torch.bmm(queries, keys.tanspose(1, 2))/ math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # attention_wieghts的shape:
        # (batch_size, 查询个数, 键值对个数)
        return torch.bmm(self.dropout(self.attention_weights),values)
        # 输出的shape:
        # (batch_size, 查询个数, 值的维度)

class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)

class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, values_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout) # 使用缩放点积注意力
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(values_size, num_hiddens, bias=bias)
        self.W_0 = nn.Linear(num_hiddens, num_hiddens, bias=bias)
    
    def forward(self, queries, keys, values, valid_lens):
        # queries, keys, values 的 形状:
        # (batch_size, 查询或者键值对的个数, query_size/key_size/values_size)
        # valid_lens 的形状:
        # (batch_size,) 或(batch_size, 查询的个数)
        # 经过变换后, 输出的queries, keys, values的形状:
        # (batch_size*num_heads, 查询或者键值对的个数, num_hiddens/num_heads)
        queries = transpose_qkv(self.self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0, 将第一项(标量或者矢量)复制num_heads次, 
            # 然后如此复制第二项, 然后诸如此类.
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

            # output的形状: (batch_size*num_heads, 查询的个数, num_hiddens/num_heads)
            output = self.attention(queries, keys, values, valid_lens)

            # 拼接
            # output_concat的形状为: (batch_size, 查询的个数, num_hiddens)
            output_concat = transpose_output(output,self.num_heads)

            return self.W_o(output_concat)
        
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.P = torch.zeros((1, max_len, num_hiddens)) # 维度要和X相同
        X = torch.arange(max_len,
                         dtype=torch.float32.reshape(-1,1)) / torch.pow(10000, 
        torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        # 该步骤相当于两层for循环

        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
    
    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)



def sequence_mask(X, valid_len, value=0):
    """
    对序列张量进行掩码操作：将每个序列中超出有效长度的位置填充为指定值
    参数：
        X: 输入张量，形状为 (batch_size, seq_len) 或 (batch_size, seq_len, feature_dim)
        valid_len: 有效长度张量，形状为 (batch_size,)，每个元素表示对应序列的有效长度
        value: 填充值，默认为0
    返回：
        掩码后的张量，形状与X完全相同
    """
    maxlen = X.size(1)
    
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :]
    
    mask = mask < valid_len.unsqueeze(1)
    
    if len(X.shape) == 3:
        mask = mask.unsqueeze(-1)  # shape=(batch_size, seq_len, 1)，广播到特征维度
    
    X = X.masked_fill(~mask, value)
    return X

def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
