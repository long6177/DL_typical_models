class Config:

    # 数据
    seq_length = 50

    # 模型
    vocab_size = None
    embedding_dim = 128
    hidden_size = 512
    num_layers = 1

    # 训练
    batch_size = 64
    lr = 0.001
    num_epochs = 20

    # 生成
    temperature = 0.8

    # 数据下载
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_path = "dataset/input.txt"
