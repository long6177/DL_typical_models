import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 输入 x_t 的线性变换
        self.Wx = nn.Linear(input_size, 4*hidden_size)

        # 上一隐藏状态 h_{t-1}
        self.Wh = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, h, c):
        """
        x : (batch, input_size)
        h : (batch, hidden_size)
        c : (batch, hidden_size)
        """
        gates = self.Wx(x) + self.Wh(h)
        """
        gates.shape = (batch, 4*hidden_size)

        里面包含： [一次性计算四个]
        input gate
        forget gate
        output gate
        candidate cell
        """
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_new = f * c + i * g

        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, x):
        batch, seq, dim = x.shape

        h = torch.zeros(batch, self.cell.hidden_size).to(x.device)
        c = torch.zeros(batch, self.cell.hidden_size).to(x.device)

        outputs = []

        for t in range(seq):
            x_t = x[:, t, :]    # (batch, input_size)
            h, c =self.cell(x_t, h, c)
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)
        """
        outputs:
        [(batch, hidden), (batch, hidden), ...]
        ==>
        (batch, seq, hidden)
        """
        return outputs, (h, c)


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layer):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        self.rnn = MyLSTM(
            embedding_dim,
            hidden_size
        )

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)      # (batch, seq)
        out, hidden = self.rnn(x) # (batch, seq, embedding_dim)
        logits = self.fc(out)   # (batch, seq, hidden_size)
        return  logits, hidden  # (batch, seq, vocab_size