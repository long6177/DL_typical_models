import torch
import torch.nn as nn

class SimpleRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.Wx = nn.Linear(input_size, hidden_size)
        self.Wh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        h_new = torch.tanh(self.Wx(x) + self.Wh(h))

        return h_new

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = SimpleRNNCell(input_size, hidden_size)

    def forward(self, x):
        batch, seq, dim = x.shape
        h = torch.zeros(batch, self.cell.hidden_size).to(x.device)

        outputs = []

        for t in range(seq): # 按时间步展开, 实现"循环"
            x_t = x[:, t, :]    # x_t.shape = (batch, embeddings_dim)
            h = self.cell(x_t, h)
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1) # 新增指定维度
        #outputs.shape: [(batch, hidden_size), (batch, hidden_size),...] ==> (batch, seq, hidden_size)
        return outputs, h

class CharRNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        # embedding层暗含权重矩阵形状为(num_embeddings, embeddings_dim)
        # 他能将接收的输入的每一个token变换成一个word vector
        # (batch, seq_len) ==> (batch, seq_len, embedding_dim)

        self.rnn = MyRNN(
            embedding_dim,
            hidden_size,
        )

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, hidden = self.rnn(x)
        logits = self.fc(out)

        return logits, hidden