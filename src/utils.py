def print_evaluation_metrics(experiment_name: str, accuracy: float, f1: float) -> None:
    # This helper function prints out the results from the model predictions on the test dataset

    print(f"{experiment_name}")
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Macro F1 : {f1 * 100:.2f}%")