from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from fake_news_detector.features import GensimVectorizer
from fake_news_detector.modeling import train_evaluate_pipeline
from fake_news_detector.parse_data import load_and_split_data
from fake_news_detector.utils import print_evaluation_metrics, save_artifacts


def main() -> None:

    train_path = "liar_dataset/train.tsv"
    test_path = "liar_dataset/test.tsv"

    x_train, y_train = load_and_split_data(train_path)
    x_test, y_test = load_and_split_data(test_path)

    vectorizer = GensimVectorizer(model_name="glove-twitter-200")
    classifier = LogisticRegression(max_iter=1000)

    # Random Forest is left as a comment, in order to enable easier switching between the two architectures
    # classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    model, test_accuracy, test_macro_f1 = train_evaluate_pipeline(
        x_train, y_train, x_test, y_test, vectorizer, classifier)

    print_evaluation_metrics("H05 - GloVe + Logistic Regression",
                             test_accuracy, test_macro_f1)

    save_artifacts({"h05_glove_logreg_pipeline.pkl": {"vectorizer": vectorizer, "classifier": model}})


if __name__ == '__main__':
    main()
