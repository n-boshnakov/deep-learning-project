from src.parse_data import load_and_split_data
from src.modeling import train_evaluate_pipeline
from src.utils import print_evaluation_metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def main() -> None:

    train_path = "liar_dataset/train.tsv"
    test_path = "liar_dataset/test.tsv"

    x_train, y_train = load_and_split_data(train_path)
    x_test, y_test = load_and_split_data(test_path)

    vectorizer = TfidfVectorizer(max_features=5000)
    classifier = LogisticRegression(max_iter=1000)

    model, test_accuracy, test_macro_f1 = train_evaluate_pipeline(x_train, y_train, x_test, y_test, vectorizer, classifier)

    print_evaluation_metrics("H02 - TF-IDF + Logistic Regression", test_accuracy, test_macro_f1)

if __name__ == '__main__':
    main()