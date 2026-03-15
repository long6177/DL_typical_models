import torch
from torch import nn
from attention import MultiHeadAttention,PositionalEncoding
import math

class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, 
    num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, 
    num_heads, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens,
            num_heads, dropout, use_bias
        )
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens
        )
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class Encoder(nn.Module):
    """基本的编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError

class TransformerEncoder(Encoder):
    """Transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size,value_size,
    num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
    num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens

        self.embedding = nn.Embedding(vocab_size, num_hiddens, dropout)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i)),
            EncoderBlock(key_size, query_size, value_size,
            num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias)

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weight[i] = blk.attention.attention.attention_weights
        
        return X

class DecoderBlock(nn.Module):
    """解码器中第i个块"""
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i

        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size,
            num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)

        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size,
            num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)

        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]

        batch_size, num_steps, _ = X.shape

        # dec_valid_lens的开头: (batch_size, num_step),
        # 其中每一行都是[1, 2,..., num_steps]
        dec_valid_lens = torch.arrange(
            1, num_steps+1, device=X.device).repeat(batch_size)

        # 自注意力
        X2 = self.attention1(X, X, X, dec_valid_lens)
        Y = self.addnorm1(X, X2)

        # 编码器-译码器注意力
        # enc_outputs的开头: (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs,
                             enc_valid_lens)
        Z = self.addnorm2(Y, Y2)

        return self.addnorm3(Z, self.ffn(Z)), state # state作为编码层的输出, 继续传递

class Decoder(nn.Module):
    """解码器基类（所有解码器的父类）"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        """
        初始化解码器的隐藏状态
        必须由子类实现，不同解码器的状态初始化逻辑不同（如Seq2Seq/注意力解码器）
        
        参数：
            enc_outputs: 编码器的输出（核心输入，解码器需基于编码器输出解码）
            *args: 其他可选参数（如编码器有效长度、批量大小等）
        返回：
            state: 解码器的初始状态（格式由子类定义，如隐藏状态、编码器输出等）
        """
        pass

    def forward(self, X, state):
        """
        解码器前向传播核心逻辑
        必须由子类实现，定义输入X和状态state如何计算输出
        
        参数：
            X: 解码器的输入序列（如目标语言的词索引序列）
            state: 解码器的状态（由init_state初始化，或上一步更新后的状态）
        返回：
            output: 解码器的输出（如词表概率分布）
            state: 更新后的解码器状态（用于下一时间步解码）
        """
        pass

class AttentionDecoder(Decoder):
    """带注意力机制的解码器基类"""
    def __init__(self, **kwargs):
        super().__init__(** kwargs)

    @property
    def attention_weights(self):
        """返回注意力权重（用于可视化）"""
        raise NotImplementedError

    def forward(self, X, state):
        """
        前向传播核心逻辑
        参数：
            X: 解码器输入，形状为 (batch_size, seq_len, num_hiddens)
            state: 解码器状态，包含：
                - 解码器上一步的隐藏状态
                - 编码器所有时间步的输出（key/value）
                - 编码器有效长度（避免对padding部分计算注意力）
        返回：
            output: 解码器输出，形状为 (batch_size, seq_len, num_hiddens)
            state: 更新后的解码器状态
        """
        enc_outputs, enc_valid_lens = state[0], state[1]
        
        X = self.embedding(X)  
        X = X.permute(1, 0, 2)
        
        outputs, attention_weights = [], []
        for x in X:
            out, state = self._one_step(x, state)
            outputs.append(out)
            attention_weights.append(self.attention_weights)
        
        outputs = torch.cat(outputs, dim=0).permute(1, 0, 2)
        return outputs, state

    def _one_step(self, x, state):
        """
        单时间步的解码逻辑（由子类实现）
        参数：
            x: 单步解码器输入，形状 (batch_size, embed_dim)
            state: 解码器状态
        返回：
            out: 单步输出，形状 (batch_size, vocab_size)
            state: 更新后的状态
        """
        raise NotImplementedError

class TransformerDecoder(AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder,self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layer = num_layers

        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)

        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                                 DecoderBlock(key_size, query_size, value_size,
                                              num_hiddens, norm_shape, ffn_num_input,
                                              ffn_num_hiddens, num_heads, dropout,i))
        self.dense = nn.Linear(num_hiddens, vocab_size)
        
    def init_state(self, enc_outputs, enc_valid_lens, *args):
        self.seqX = None
        return [enc_outputs, enc_valid_lens]

    def forward(self, X, state):
        if not self.training:
            self.seqX = X if self.seqX is None else torch.cat((self.seqX, X), dim=1)

        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        
        if not self.training:
            return self.dense(X)[:, -1:, :], state
        
        return self.dense(X), state

    def attention_weights(self):
        return self._attention_weights





class PositionWiseFFN(nn.Block):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Dense(ffn_num_hiddens, flatten=False,
                               activation='relu')
        self.dense2 = nn.Dense(ffn_num_outputs, flatten=False)

    def forward(self, X):
        return self.dense2(self.dense1(X))

class AddNorm(nn.Block):
    """残差连接后进行层规范化"""
    def __init__(self, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm()

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
