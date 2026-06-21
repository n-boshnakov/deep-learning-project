import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fake_news_detector.modeling import train_evaluate_pytorch_model
from fake_news_detector.parse_data import LiarDataset, build_vocab, load_and_split_data
from fake_news_detector.utils import plot_training_history, print_evaluation_metrics

MAX_VOCAB_SIZE = 15000
MAX_SEQ_LEN = 50
EMBEDDING_DIM = 50
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001

try:
    import torch_directml
    device = torch_directml.device()
except ImportError:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaselineEmbeddingNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(BaselineEmbeddingNet, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)
        return self.fc(pooled)

def main():
    train_path = os.path.join("liar_dataset", "train.tsv")
    test_path = os.path.join("liar_dataset", "test.tsv")

    x_train, y_train = load_and_split_data(train_path)
    x_test, y_test = load_and_split_data(test_path)

    word2idx = build_vocab(x_train, max_vocab_size=MAX_VOCAB_SIZE)

    train_dataset = LiarDataset(x_train, y_train, word2idx, max_seq_len=MAX_SEQ_LEN)
    test_dataset = LiarDataset(x_test, y_test, word2idx, max_seq_len=MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = BaselineEmbeddingNet(len(word2idx), EMBEDDING_DIM, 6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trained_model, macro_f1, macro_prec, history = train_evaluate_pytorch_model(
        model, train_loader, test_loader, criterion, optimizer, device, epochs=EPOCHS
    )

    print()
    print_evaluation_metrics("H06 - PyTorch Trainable Embeddings", macro_f1, macro_prec)

    plot_training_history(history, "H06_Embeddings")

if __name__ == "__main__":
    main()