import os
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from fake_news_detector.modeling import HybridBertFakeNewsNet
from fake_news_detector.parse_data import LiarHybridBertDataset, MetadataPreprocessor
from fake_news_detector.utils import (plot_training_history, print_evaluation_metrics,
                                      save_artifacts)

MODEL_NAME = 'bert-base-uncased'
MAX_SEQ_LEN = 64
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 2e-5
MIN_CATEGORY_FREQUENCY = 10
DROPOUT_RATE = 0.6
WEIGHT_DECAY = 0.01

# Mac Hardware Acceleration (Apple Silicon / Metal Performance Shaders)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def main():
    print(f"Hardware Acceleration: {device}")
    print(f"Active Architecture: Hybrid BERT (Text + Preprocessed Metadata)")

    train_path = os.path.join("liar_dataset", "train.tsv")
    test_path = os.path.join("liar_dataset", "test.tsv")

    cols = [
        "id", "label", "statement", "subjects", "speaker", "speaker_job",
        "state_info", "party_affiliation", "barely_true_counts",
        "false_counts", "half_true_counts", "mostly_true_counts",
        "pants_on_fire_counts", "context"
    ]
    df_train = pd.read_csv(train_path,
                           sep='\t',
                           header=None,
                           names=cols,
                           quoting=3).dropna(subset=['statement', 'label'])
    df_test = pd.read_csv(test_path,
                          sep='\t',
                          header=None,
                          names=cols,
                          quoting=3).dropna(subset=['statement', 'label'])

    num_cols = [
        "barely_true_counts", "false_counts", "half_true_counts",
        "mostly_true_counts", "pants_on_fire_counts"
    ]
    for col in num_cols:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(0)
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce').fillna(0)

    cat_cols = [
        "subjects", "speaker", "speaker_job", "state_info",
        "party_affiliation", "context"
    ]
    df_train[cat_cols] = df_train[cat_cols].fillna("unknown")
    df_test[cat_cols] = df_test[cat_cols].fillna("unknown")

    label_map = {
        "pants-fire": 0,
        "false": 1,
        "barely-true": 2,
        "half-true": 3,
        "mostly-true": 4,
        "true": 5
    }
    df_train['label_idx'] = df_train['label'].map(label_map)
    df_test['label_idx'] = df_test['label'].map(label_map)

    preprocessor = MetadataPreprocessor(min_frequency=MIN_CATEGORY_FREQUENCY)
    preprocessor.fit(df_train)

    train_meta_tensor = preprocessor.transform(df_train)
    test_meta_tensor = preprocessor.transform(df_test)
    meta_feature_count = train_meta_tensor.shape[1]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = LiarHybridBertDataset(df_train['statement'].values,
                                          train_meta_tensor,
                                          df_train['label_idx'].values,
                                          tokenizer, MAX_SEQ_LEN)
    test_dataset = LiarHybridBertDataset(df_test['statement'].values,
                                         test_meta_tensor,
                                         df_test['label_idx'].values,
                                         tokenizer, MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

    model = HybridBertFakeNewsNet(bert_model_name=MODEL_NAME,
                                  meta_dim=meta_feature_count,
                                  dropout_rate=DROPOUT_RATE)
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
            meta_features = batch['meta_features'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           meta_input=meta_features)
            loss = criterion(logits, labels)

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
                meta_features = batch['meta_features'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               meta_input=meta_features)
                loss = criterion(logits, labels)
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
    print_evaluation_metrics("H13 - Hybrid BERT Updated",
                             history['val_f1'][-1], final_macro_prec)
    plot_training_history(history, "H13_Hybrid_BERT_Updated")

    artifacts_to_save = {
        "h13_hybrid_bert_updated_weights.pth": model,
        "h13_metadata_preprocessor.pkl": preprocessor
    }
    save_artifacts(artifacts_to_save)


if __name__ == "__main__":
    main()
