import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, precision_score
from torch.utils.data import DataLoader


def train_evaluate_pipeline(
        x_train: pd.Series, y_train: pd.Series, x_test: pd.Series,
        y_test: pd.Series, vectorizer: BaseEstimator,
        classifier: BaseEstimator) -> tuple[BaseEstimator, float, float]:

    vectorized_x_train = vectorizer.fit_transform(x_train)
    vectorized_x_test = vectorizer.transform(x_test)

    model = classifier.fit(vectorized_x_train, y_train)

    predictions = classifier.predict(vectorized_x_test)

    test_macro_f1 = f1_score(y_test, predictions, average="macro")
    test_macro_precision = precision_score(y_test,
                                           predictions,
                                           average="macro",
                                           zero_division=0)

    return model, test_macro_f1, test_macro_precision

def evaluate_model(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_true: list[int] = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()

            _, predicted_classes = torch.max(predictions, 1)
            all_preds.extend(predicted_classes.cpu().numpy())
            all_true.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    macro_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    macro_prec = precision_score(all_true, all_preds, average="macro", zero_division=0)

    return avg_loss, macro_f1, macro_prec

def train_evaluate_pytorch_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device, 
    epochs: int = 10
) -> tuple[nn.Module, float, float, dict]:
    
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        all_train_preds: list[int] = []
        all_train_true: list[int] = []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            _, predicted_classes = torch.max(predictions, 1)
            all_train_preds.extend(predicted_classes.cpu().numpy())
            all_train_true.extend(batch_y.cpu().numpy())
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_f1 = f1_score(all_train_true, all_train_preds, average="macro", zero_division=0)
        
        avg_val_loss, val_f1, val_prec = evaluate_model(model, test_loader, criterion, device)
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
            
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")
        
    return model, history["val_f1"][-1], val_prec, history

class BaselineEmbeddingNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(BaselineEmbeddingNet, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)
        return self.fc(pooled)