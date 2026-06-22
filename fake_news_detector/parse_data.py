from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
    
def load_hybrid_data(data_file: str) -> pd.DataFrame:
    columns = [
        "id", "label", "statement", "subjects", "speaker", 
        "speaker_job", "state_info", "party_affiliation", 
        "barely_true_counts", "false_counts", "half_true_counts", 
        "mostly_true_counts", "pants_on_fire_counts", "context"
    ]
    df = pd.read_csv(data_file, sep="\t", header=None, names=columns, quoting=3)
    
    df = df.dropna(subset=["label", "statement"])
    
    num_cols = ["barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts"]
    df[num_cols] = df[num_cols].fillna(0.0)

    cat_cols = ["subjects", "speaker", "speaker_job", "state_info", "party_affiliation", "context"]
    df[cat_cols] = df[cat_cols].fillna("unknown")
    
    return df

class MetadataPreprocessor:
    """Handles scaling of numerical columns and one-hot encoding of categorical columns."""
    def __init__(self, min_frequency: int = 1):
        self.num_cols = [
            "barely_true_counts", "false_counts", "half_true_counts", 
            "mostly_true_counts", "pants_on_fire_counts"
        ]
        self.cat_cols = [
            "party_affiliation", "state_info", "speaker_job", 
            "speaker", "context", "subjects"
        ]
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=min_frequency), self.cat_cols)
            ]
        )
        
    def fit(self, df: pd.DataFrame):
        """Learns the vocabulary of categories and the mean/std of numbers from the TRAIN set."""
        self.preprocessor.fit(df)
        
    def transform(self, df: pd.DataFrame) -> torch.Tensor:
        """Transforms any DataFrame into a numerical PyTorch tensor."""
        features = self.preprocessor.transform(df)
        return torch.tensor(features, dtype=torch.float32)

class LiarHybridDataset(Dataset):
    def __init__(self, df: pd.DataFrame, metadata_tensor: torch.Tensor, word2idx: dict, max_seq_len: int = 50):
        self.x_text = text_to_indices(df["statement"], word2idx, max_seq_len)
        
        self.x_meta = metadata_tensor
        
        self.label_map = {
            "pants-fire": 0, "false": 1, "barely-true": 2, 
            "half-true": 3, "mostly-true": 4, "true": 5
        }
        self.y = torch.tensor([self.label_map[label] for label in df["label"]], dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.x_text[idx], self.x_meta[idx], self.y[idx]