<h1 align='center'> 简单rnn实现字符级语言模型 </h1> 

## 一. 结构

### 1. shape变化
| 层 | 输出 |
| :-: |----|
|输入|(batch, seq_len)
|embedding|(batch, seq_len, embed_dim)|
|RNN|(batch, seq_len, hidden_dim)|
|Linear输出|(batch, seq_len, vocab_size)|