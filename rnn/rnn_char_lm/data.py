import os
import torch
from torch.utils.data import Dataset
import requests

def download_data(url, path):
    if os.path.exists(path):
        return
    
    print("Downloading dataset...")

    r = requests.get(url)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding='utf-8') as f:
        f.write(r.text)

    print("Download complete")

class CharDataset(Dataset):
    def __init__(self, text_ids, seq_length):
        self.data = text_ids
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+1:idx+self.seq_length+1]

        return torch.tensor(x), torch.tensor(y)

if __name__ == "__main__":
    from text_utils import CharTokenizer
    from config import Config

    config  = Config()

    download_data(config.data_url, config.data_path)

    with open(config.data_path, "r", encoding="utf-8") as f:
        text = f.read()
    text = text[:20000]

    tokenizer = CharTokenizer(text)
    text_ids = tokenizer.encode(text)
    text_ids

    config.vocab_size = tokenizer.vocab_size
    vocab = list(tokenizer.stoi.keys())
    print("vocab_size: ",config.vocab_size)
    print("vocab: ", vocab)
    print("text_ids.shape:",'('+str(len(text_ids))+')')

    dataset = CharDataset(text_ids, config.seq_length)
    
    print("len(dataset):",len(dataset))
    input, target = dataset[0]
    print("input:  ",input)
    print("target: ",target)