import os

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fake_news_detector.modeling import HybridRNNFakeNewsNet, train_evaluate_pytorch_model
from fake_news_detector.parse_data import (LiarHybridDataset, MetadataPreprocessor, build_vocab,
                                           load_hybrid_data)
from fake_news_detector.utils import (plot_training_history, print_evaluation_metrics,
                                      save_artifacts)

MAX_VOCAB_SIZE = 15000
MAX_SEQ_LEN = 50
EMBEDDING_DIM = 50
RNN_HIDDEN_DIM = 128
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.0005
DROPOUT_RATE = 0.4
MIN_CATEGORY_FREQUENCY = 10

RNN_TYPE = 'GRU'

# try:
#     import torch_directml
#     device = torch_directml.device()
# except ImportError:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")


def main():
    print(f"Using device: {device}")
    print(f"Active Architecture: Hybrid {RNN_TYPE}")

    train_path = os.path.join("liar_dataset", "train.tsv")
    test_path = os.path.join("liar_dataset", "test.tsv")

    df_train = load_hybrid_data(train_path)
    df_test = load_hybrid_data(test_path)

    print("Fitting metadata preprocessor...")
    preprocessor = MetadataPreprocessor(min_frequency=MIN_CATEGORY_FREQUENCY)

    preprocessor.fit(df_train)

    train_meta_tensor = preprocessor.transform(df_train)
    test_meta_tensor = preprocessor.transform(df_test)

    meta_feature_count = train_meta_tensor.shape[1]
    print(
        f"Metadata transformed into {meta_feature_count} numerical features per statement."
    )

    word2idx = build_vocab(df_train["statement"],
                           max_vocab_size=MAX_VOCAB_SIZE)

    train_dataset = LiarHybridDataset(df_train,
                                      train_meta_tensor,
                                      word2idx,
                                      max_seq_len=MAX_SEQ_LEN)
    test_dataset = LiarHybridDataset(df_test,
                                     test_meta_tensor,
                                     word2idx,
                                     max_seq_len=MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

    model = HybridRNNFakeNewsNet(vocab_size=len(word2idx),
                                 embed_dim=EMBEDDING_DIM,
                                 hidden_dim=RNN_HIDDEN_DIM,
                                 meta_dim=meta_feature_count,
                                 num_classes=6,
                                 rnn_type=RNN_TYPE,
                                 dropout_rate=DROPOUT_RATE).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE,
                           weight_decay=1e-4)

    print(f"\nStarting Training for Hybrid {RNN_TYPE} Model...")
    trained_model, macro_f1, macro_prec, history = train_evaluate_pytorch_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        epochs=EPOCHS)

    print()
    print_evaluation_metrics(f"H08 - Hybrid {RNN_TYPE} (Text + Metadata)",
                             macro_f1, macro_prec)

    plot_training_history(history, f"H09_Hybrid_{RNN_TYPE}_Improved-2")

    # artifacts_to_save = {
    #     f"h08_hybrid_{RNN_TYPE.lower()}_weights.pth": trained_model,
    #     f"h08_{RNN_TYPE.lower()}_word2idx.pkl": word2idx,
    #     "metadata_preprocessor.pkl": preprocessor
    # }

    # save_artifacts(artifacts_to_save)


if __name__ == "__main__":
    main()
