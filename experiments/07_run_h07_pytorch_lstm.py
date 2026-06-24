import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fake_news_detector.modeling import train_evaluate_pytorch_model
from fake_news_detector.parse_data import LiarDataset, build_vocab, load_and_split_data
from fake_news_detector.utils import plot_training_history, print_evaluation_metrics, save_artifacts

MAX_VOCAB_SIZE = 15000
MAX_SEQ_LEN = 50
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001

# Force CPU execution because DirectML does not currently support PyTorch's fused LSTM operations
device = torch.device("cpu")
print("Using CPU for LSTM compatibility...")


class LSTMNet(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(LSTMNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim,
                            hidden_dim,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)

        hidden_fwd = hidden[-2, :, :]
        hidden_bwd = hidden[-1, :, :]
        hidden_cat = torch.cat((hidden_fwd, hidden_bwd), dim=1)

        dropped = self.dropout(hidden_cat)
        return self.fc(dropped)


def main():
    train_path = os.path.join("liar_dataset", "train.tsv")
    test_path = os.path.join("liar_dataset", "test.tsv")

    x_train, y_train = load_and_split_data(train_path)
    x_test, y_test = load_and_split_data(test_path)

    word2idx = build_vocab(x_train, MAX_VOCAB_SIZE)
    vocab_size = len(word2idx)

    train_dataset = LiarDataset(x_train, y_train, word2idx, MAX_SEQ_LEN)
    test_dataset = LiarDataset(x_test, y_test, word2idx, MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

    model = LSTMNet(vocab_size, EMBEDDING_DIM, HIDDEN_DIM,
                    num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trained_model, macro_f1, macro_prec, history = train_evaluate_pytorch_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        epochs=EPOCHS)

    print()
    print_evaluation_metrics("H07 - PyTorch Bidirectional LSTM", macro_f1,
                             macro_prec)

    plot_training_history(history, "H07_LSTM")

    artifacts_to_save = {
        "h07_lstm_weights.pth": trained_model,
        "h07_lstm_word2idx.pkl": word2idx
    }
    save_artifacts(artifacts_to_save)


if __name__ == "__main__":
    main()
