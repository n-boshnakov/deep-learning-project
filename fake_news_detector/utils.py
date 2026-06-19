def print_evaluation_metrics(experiment_name: str, f1: float, precision: float) -> None:

    print(f"Experiment: {experiment_name}")
    print(f"Macro F1: {f1 * 100:.2f}%")
    print(f"Macro Precision: {precision * 100:.2f}%")