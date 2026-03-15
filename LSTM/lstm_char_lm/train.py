import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader

class Trainer:

    def __init__(self, model, dataset, config, device):
        self.model = model.to(device)

        self.loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True
        )

        self.device = device

        self.critetion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr
        )

        self.config = config

        self.save_path = 'saved_models/temp_params.pt'

    def train(self):
        best_loss = float("inf")

        self.model.train()
        for epoch in range(self.config.num_epochs):
            total_loss = 0

            for x, y in self.loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits, _ = self.model(x)

                loss = self.critetion(
                    # 每一个时间步预测出真实值的期望的平均交叉熵
                    logits.reshape(-1, logits.size(-1)), # (batch*seq, vocab_size)
                    y.reshape(-1) # (batch*seq)
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
            
            ave_loss = total_loss / len(self.loader)
            print("epoch:", epoch+1,f"Loss: {ave_loss:.4f}")

            if ave_loss < best_loss:
                best_loss = ave_loss
                os.makedirs("saved_models",exist_ok=True)
                torch.save(self.model.state_dict(),self.save_path)


if __name__ == "__main__":
    from config import Config
    from text_utils import CharTokenizer
    from data import download_data, CharDataset
    from model import CharLSTM
    
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    download_data(config.data_url, config.data_path)
    with open(config.data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    text = text[:60000]
    tokenizer = CharTokenizer(text)
    config.vocab_size = tokenizer.vocab_size
    text_ids = tokenizer.encode(text)

    dataset = CharDataset(text_ids, config.seq_length)

    model = CharLSTM(config.vocab_size,
                     config.embedding_dim,
                     config.hidden_size,
                     config.num_layers)

    trainer = Trainer(model, dataset, config, device)
    trainer.train()