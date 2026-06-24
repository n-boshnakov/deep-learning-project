import os
import time

import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fake_news_detector.parse_data import LABEL_MAP, LiarTransformerDataset
from fake_news_detector.utils import (plot_training_history, print_evaluation_metrics,
                                      save_artifacts)

MODEL_NAME = 'distilbert-base-uncased'
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
    print(f"Active Architecture: {MODEL_NAME} (Text-Only)")

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
    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
    total_start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()

        model.train()
        total_train_loss = 0
        train_preds_epoch = []
        train_labels_epoch = []

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            train_preds_epoch.extend(preds.detach().cpu().numpy())
            train_labels_epoch.extend(labels.cpu().numpy())

        avg_train_loss = total_train_loss / len(train_loader)
        train_f1 = f1_score(train_labels_epoch,
                            train_preds_epoch,
                            average='macro')

        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                total_val_loss += outputs.loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(test_loader)
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        epoch_time = time.time() - epoch_start_time
        mins, secs = divmod(int(epoch_time), 60)
        print(
            f"Epoch [{epoch}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | Time: {mins}m {secs}s"
        )

    total_time = time.time() - total_start_time
    total_mins, total_secs = divmod(int(total_time), 60)

    final_macro_prec = precision_score(all_labels,
                                       all_preds,
                                       average='macro',
                                       zero_division=0)

    print(f"\nTotal Training Time: {total_mins}m {total_secs}s")
    print_evaluation_metrics(f"H11 - {MODEL_NAME}", history['val_f1'][-1],
                             final_macro_prec)
    plot_training_history(history, f"H11_distilbert")

    save_artifacts({"h11_distilbert_weights.pth": model})


if __name__ == "__main__":
    main()
