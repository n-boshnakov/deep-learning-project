from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset


def load_and_split_data(data_file: str) -> tuple[pd.Series, pd.Series]:
    columns = [
        "id", "label", "statement", "subjects", "speaker", 
        "speaker_job", "state_info", "party_affiliation", 
        "barely_true_counts", "false_counts", "half_true_counts", 
        "mostly_true_counts", "pants_on_fire_counts", "context"
    ]
    loaded_data = pd.read_csv(data_file, sep="\t", header=None, names=columns, quoting=3)
    data_clean = loaded_data.dropna(subset=["label", "statement"])
    return data_clean["statement"], data_clean["label"]

def build_vocab(texts: pd.Series, max_vocab_size: int = 15000) -> dict:
    all_words = []
    for text in texts:
        all_words.extend(text.lower().split())
    
    counter = Counter(all_words)
    most_common = counter.most_common(max_vocab_size)
    
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for idx, (word, _) in enumerate(most_common, start=2):
        word2idx[word] = idx
        
    return word2idx

def text_to_indices(texts: pd.Series, word2idx: dict, max_seq_len: int = 50) -> torch.Tensor:
    sequences = []
    for text in texts:
        words = text.lower().split()
        seq = [word2idx.get(w, word2idx["<UNK>"]) for w in words]
        
        if len(seq) > max_seq_len:
            seq = seq[:max_seq_len]
        else:
            seq = seq + [0] * (max_seq_len - len(seq))
            
        sequences.append(seq)
    return torch.tensor(sequences, dtype=torch.long)

class LiarDataset(Dataset):
    def __init__(self, x: pd.Series, y: pd.Series, word2idx: dict, max_seq_len: int = 50):
        self.x = text_to_indices(x, word2idx, max_seq_len)
        self.label_map = {
            "pants-fire": 0, "false": 1, "barely-true": 2, 
            "half-true": 3, "mostly-true": 4, "true": 5
        }
        self.y = torch.tensor([self.label_map[label] for label in y], dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]