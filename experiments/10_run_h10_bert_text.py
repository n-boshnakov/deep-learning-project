import os

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fake_news_detector.modeling import train_evaluate_transformer_model
from fake_news_detector.parse_data import LABEL_MAP, LiarTransformerDataset
from fake_news_detector.utils import (plot_training_history, print_evaluation_metrics,
                                      save_artifacts)

MODEL_NAME = 'bert-base-uncased'
MAX_SEQ_LEN = 64
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 2e-5

# Mac Hardware Acceleration (Apple Silicon / Metal Performance Shaders)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def main():
    print(f"Hardware Acceleration: {device}")
    print(f"Active Architecture: {MODEL_NAME} (Text-Only Baseline)")

    train_path = os.path.join("liar_dataset", "train.tsv")
    test_path = os.path.join("liar_dataset", "test.tsv")

    cols = [
        "id", "label", "statement", "subject", "speaker", "job", "state",
        "party", "barely", "false", "half", "mostly", "pants", "context"
    ]
    df_train = pd.read_csv(train_path, sep='\t', header=None,
                           names=cols).dropna(subset=['statement', 'label'])
    df_test = pd.read_csv(test_path, sep='\t', header=None,
                          names=cols).dropna(subset=['statement', 'label'])

    df_train['label_idx'] = df_train['label'].map(LABEL_MAP)
    df_test['label_idx'] = df_test['label'].map(LABEL_MAP)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
                                                               num_labels=6)
    model.to(device)

    train_dataset = LiarTransformerDataset(df_train['statement'].values,
                                           df_train['label_idx'].values,
                                           tokenizer, MAX_SEQ_LEN)
    test_dataset = LiarTransformerDataset(df_test['statement'].values,
                                          df_test['label_idx'].values,
                                          tokenizer, MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"\nStarting Fine-Tuning for {EPOCHS} Epochs...")

    model, val_f1, final_macro_prec, history = train_evaluate_transformer_model(
        model, train_loader, test_loader, optimizer, device, epochs=EPOCHS)

    print_evaluation_metrics(f"H10 - BERT Text-Only", val_f1, final_macro_prec)
    plot_training_history(history, "H10_BERT_Text")
    save_artifacts({"h10_bert_text_weights.pth": model})


if __name__ == "__main__":
    main()
