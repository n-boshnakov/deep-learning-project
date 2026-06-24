from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset

LABEL_MAP: dict[str, int] = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5
}

LIAR_COLUMNS: list[str] = [
    "id", "label", "statement", "subjects", "speaker", "speaker_job",
    "state_info", "party_affiliation", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
]

NUMERIC_COLS: list[str] = [
    "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts"
]

CAT_COLS: list[str] = [
    "subjects", "speaker", "speaker_job", "state_info", "party_affiliation",
    "context"
]


def load_and_split_data(data_file: str) -> tuple[pd.Series, pd.Series]:
    loaded_data = pd.read_csv(data_file,
                              sep="\t",
                              header=None,
                              names=LIAR_COLUMNS,
                              quoting=3)
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


def text_to_indices(texts: pd.Series,
                    word2idx: dict,
                    max_seq_len: int = 50) -> torch.Tensor:
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

    def __init__(self,
                 x: pd.Series,
                 y: pd.Series,
                 word2idx: dict,
                 max_seq_len: int = 50):
        self.x = text_to_indices(x, word2idx, max_seq_len)
        self.y = torch.tensor([LABEL_MAP[label] for label in y],
                              dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_hybrid_data(data_file: str) -> pd.DataFrame:
    df = pd.read_csv(data_file,
                     sep="\t",
                     header=None,
                     names=LIAR_COLUMNS,
                     quoting=3)
    df = df.dropna(subset=["label", "statement"])
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0.0)
    df[CAT_COLS] = df[CAT_COLS].fillna("unknown")
    return df


class MetadataPreprocessor:

    def __init__(self, min_frequency: int = 1):
        self.num_cols = NUMERIC_COLS
        self.cat_cols = [
            "party_affiliation", "state_info", "speaker_job", "speaker",
            "context", "subjects"
        ]

        self.preprocessor = ColumnTransformer(
            transformers=[('num', StandardScaler(), self.num_cols),
                          ('cat',
                           OneHotEncoder(handle_unknown='ignore',
                                         sparse_output=False,
                                         min_frequency=min_frequency),
                           self.cat_cols)])

    def fit(self, df: pd.DataFrame):
        self.preprocessor.fit(df)

    def transform(self, df: pd.DataFrame) -> torch.Tensor:
        features = self.preprocessor.transform(df)
        return torch.tensor(features, dtype=torch.float32)


class LiarHybridDataset(Dataset):

    def __init__(self,
                 df: pd.DataFrame,
                 metadata_tensor: torch.Tensor,
                 word2idx: dict,
                 max_seq_len: int = 50):
        self.x_text = text_to_indices(df["statement"], word2idx, max_seq_len)
        self.x_meta = metadata_tensor
        self.y = torch.tensor([LABEL_MAP[label] for label in df["label"]],
                              dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_text[idx], self.x_meta[idx], self.y[idx]


class LiarTransformerDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class LiarHybridBertDataset(Dataset):

    def __init__(self, texts, meta_features, labels, tokenizer, max_len):
        self.texts = texts
        self.meta_features = meta_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        meta = self.meta_features[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'meta_features': meta,
            'labels': torch.tensor(label, dtype=torch.long)
        }
