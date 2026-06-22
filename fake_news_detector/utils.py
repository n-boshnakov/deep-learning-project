import os
import pickle
from typing import Any

import joblib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def print_evaluation_metrics(experiment_name: str, f1: float,
                             precision: float) -> None:

    print(f"Experiment: {experiment_name}")
    print(f"Macro Precision: {precision * 100:.2f}%")
    print(f"Macro F1: {f1 * 100:.2f}%")
 
def save_artifacts(artifacts: dict[str, Any], base_dir: str = "models") -> None:
    os.makedirs(base_dir, exist_ok=True)
    print()
    
    for file_name, obj in artifacts.items():
        save_path = os.path.join(base_dir, file_name)

        # 1. If it's a PyTorch Neural Network, save its state dictionary
        if isinstance(obj, nn.Module):
            torch.save(obj.state_dict(), save_path)
            
        # 2. If it's a Python dictionary (like our word2idx vocabulary), use pickle
        elif isinstance(obj, dict):
            with open(save_path, 'wb') as f:
                pickle.dump(obj, f)
                
        # 3. For everything else (Scikit-Learn pipelines, preprocessors), use joblib
        else:
            joblib.dump(obj, save_path)

        print(f"Successfully saved artifact under: \"{save_path}\"")

def plot_training_history(history: dict, experiment_name: str, base_dir: str = "plots") -> None:
    os.makedirs(base_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], 'b-', label='Training Loss')
    plt.plot(epochs, history["val_loss"], 'r-', label='Validation Loss')
    plt.title(f'{experiment_name} - Loss (Bias-Variance)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_f1"], 'b-', label='Training F1')
    plt.plot(epochs, history["val_f1"], 'r-', label='Validation F1')
    plt.title(f'{experiment_name} - Macro F1')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(base_dir, f"{experiment_name.replace(' ', '_')}_history.png")
    plt.savefig(save_path)
    plt.close()