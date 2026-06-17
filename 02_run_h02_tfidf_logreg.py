from src.parse_data import load_and_split_data
from src.modeling import train_evaluate_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def main() -> None:

    train_path = "liar_dataset/train.tsv"
    test_path = "liar_dataset/test.tsv"

    x_train, y_train = load_and_split_data(train_path)
    x_test, y_test = load_and_split_data(test_path)

    vectorizer = TfidfVectorizer(max_features=5000)
    classifier = LogisticRegression(max_iter=1000)

    results = train_evaluate_pipeline(x_train, y_train, x_test, y_test, vectorizer, classifier)

    print(f"H02 - TF-IDF + Logistic Regression")
    print(f"Accuracy: {results[1] * 100:.2f}%")
    print(f"Macro F1: {results[2] * 100:.2f}%")

if __name__ == '__main__':
    main()