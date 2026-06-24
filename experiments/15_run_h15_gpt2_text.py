import os
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fake_news_detector.parse_data import LABEL_MAP, LIAR_COLUMNS
from fake_news_detector.utils import (plot_training_history, print_evaluation_metrics,
                                      save_artifacts)

MODEL_NAME = 'gpt2'
MAX_SEQ_LEN = 64
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.1

# Mac Hardware Acceleration
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class LiarGptDataset(Dataset):

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

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
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


def main():
    print(f"Hardware Acceleration: {device}")
    print(f"Active Architecture: GPT-2 Text-Only Classification")

    train_path = os.path.join("liar_dataset", "train.tsv")
    test_path = os.path.join("liar_dataset", "test.tsv")

    df_train = pd.read_csv(train_path,
                           sep='\t',
                           header=None,
                           names=LIAR_COLUMNS,
                           quoting=3).dropna(subset=['statement', 'label'])
    df_test = pd.read_csv(test_path,
                          sep='\t',
                          header=None,
                          names=LIAR_COLUMNS,
                          quoting=3).dropna(subset=['statement', 'label'])

    df_train['label_idx'] = df_train['label'].map(LABEL_MAP)
    df_test['label_idx'] = df_test['label'].map(LABEL_MAP)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = LiarGptDataset(df_train['statement'].values,
                                   df_train['label_idx'].values, tokenizer,
                                   MAX_SEQ_LEN)
    test_dataset = LiarGptDataset(df_test['statement'].values,
                                  df_test['label_idx'].values, tokenizer,
                                  MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=6,
        resid_pdrop=DROPOUT_RATE,
        attn_pdrop=DROPOUT_RATE,
        embd_pdrop=DROPOUT_RATE,
        summary_first_dropout=DROPOUT_RATE)
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)

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
            logits = outputs.logits

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
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
                loss = outputs.loss
                logits = outputs.logits
                total_val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
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
    print_evaluation_metrics("H15 - GPT-2 Base", history['val_f1'][-1],
                             final_macro_prec)
    plot_training_history(history, "H15_GPT2_Base")

    save_artifacts({"h15_gpt2_base_weights.pth": model})


if __name__ == "__main__":
    main()
