import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from fake_news_detector.modeling import HybridBertFakeNewsNet, train_evaluate_transformer_model
from fake_news_detector.parse_data import (CAT_COLS, LABEL_MAP, LIAR_COLUMNS, NUMERIC_COLS,
                                           LiarHybridBertDataset, MetadataPreprocessor)
from fake_news_detector.utils import (plot_training_history, print_evaluation_metrics,
                                      save_artifacts)

MODEL_NAME = 'bert-base-uncased'
MAX_SEQ_LEN = 64
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 2e-5
MIN_CATEGORY_FREQUENCY = 10

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

    for col in NUMERIC_COLS:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(0)
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce').fillna(0)

    df_train[CAT_COLS] = df_train[CAT_COLS].fillna("unknown")
    df_test[CAT_COLS] = df_test[CAT_COLS].fillna("unknown")

    df_train['label_idx'] = df_train['label'].map(LABEL_MAP)
    df_test['label_idx'] = df_test['label'].map(LABEL_MAP)

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
                                  meta_dim=meta_feature_count)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model, val_f1, final_macro_prec, history = train_evaluate_transformer_model(
        model,
        train_loader,
        test_loader,
        optimizer,
        device,
        epochs=EPOCHS,
        criterion=criterion)

    print_evaluation_metrics("H12 - Hybrid BERT", val_f1, final_macro_prec)
    plot_training_history(history, "H12_Hybrid_BERT")
    artifacts_to_save = {
        "h12_hybrid_bert_weights.pth": model,
        "h12_metadata_preprocessor.pkl": preprocessor
    }
    save_artifacts(artifacts_to_save)


if __name__ == "__main__":
    main()
