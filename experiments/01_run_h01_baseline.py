import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, precision_score

from fake_news_detector.utils import print_evaluation_metrics


def main() -> None:
    columns = [
        "id", "label", "statement", "subjects", "speaker", "speaker_job",
        "state_info", "party_affiliation", "barely_true_counts",
        "false_counts", "half_true_counts", "mostly_true_counts",
        "pants_on_fire_counts", "context"
    ]

    train_df = pd.read_csv("liar_dataset/train.tsv",
                           sep="\t",
                           header=None,
                           names=columns,
                           quoting=3)
    test_df = pd.read_csv("liar_dataset/test.tsv",
                          sep="\t",
                          header=None,
                          names=columns,
                          quoting=3)

    # Cleanining up the data from rows missing crucial values
    train_df = train_df.dropna(subset=["label", "statement"])
    test_df = test_df.dropna(subset=["label", "statement"])

    X_train, y_train = train_df["statement"], train_df["label"]
    X_test, y_test = test_df["statement"], test_df["label"]

    # Initializing and "training" the DummyClassifier
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)

    # Generating predictions
    baseline_predictions = dummy_clf.predict(X_test)

    # Collecting test metrics
    test_macro_f1 = f1_score(y_test, baseline_predictions, average="macro")
    test_macro_precision = precision_score(y_test,
                                           baseline_predictions,
                                           average="macro",
                                           zero_division=0)

    print_evaluation_metrics("H01 - baseline", test_macro_precision,
                             test_macro_f1)


if __name__ == '__main__':
    main()
