from fake_news_detector.parse_data import load_and_split_data
from fake_news_detector.modeling import train_evaluate_pipeline
from fake_news_detector.utils import print_evaluation_metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def main() -> None:

    train_path = "liar_dataset/train.tsv"
    test_path = "liar_dataset/test.tsv"

    x_train, y_train = load_and_split_data(train_path)
    x_test, y_test = load_and_split_data(test_path)

    vectorizer = TfidfVectorizer(max_features=4500, stop_words='english', ngram_range=(1, 2))
    classifier = LogisticRegression(max_iter=900)

    model, test_accuracy, test_macro_f1 = train_evaluate_pipeline(x_train, y_train, x_test, y_test, vectorizer, classifier)

    print_evaluation_metrics("H03 - TF-IDF (N-grams + StopWords) + LogReg", test_accuracy, test_macro_f1)

if __name__ == '__main__':
    main()