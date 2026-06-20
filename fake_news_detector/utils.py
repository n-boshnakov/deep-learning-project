import os
import joblib
from typing import Any

def print_evaluation_metrics(experiment_name: str, f1: float,
                             precision: float) -> None:

    print(f"Experiment: {experiment_name}")
    print(f"Macro Precision: {precision * 100:.2f}%")
    print(f"Macro F1: {f1 * 100:.2f}%")
 
def save_model_pipeline(vectorizer: Any, classifier: Any, file_name: str, base_dir: str = "models") -> None:
    # Saves the NLP vectorizer and the machine learning classifier together as a pipeline in a .pkl file.

    os.makedirs(base_dir, exist_ok=True)
    
    save_path = os.path.join(base_dir, file_name)

    pipeline_data = {
        "vectorizer": vectorizer,
        "classifier": classifier
    }
    
    joblib.dump(pipeline_data, save_path)