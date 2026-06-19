import unittest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from fake_news_detector import modeling

class TestTrainEvaluatePipeline(unittest.TestCase):

    def test_when_valid_data_passed_then_returns_model_and_float_metrics(self):
        # Arrange
        x_train = ["this is test data", "this text contains only numbers", "this is real data", "this statement is true"]
        y_train = ["true", "false", "false", "true"]

        x_test = ["this statement is false", "123456"]
        y_test = ["false", "true"]

        vectorizer = TfidfVectorizer()
        classifier = LogisticRegression()

        expected_metric_type = float
        expected_min_value = 0.0
        expected_max_value = 1.0

        # Act
        actual_model, actual_acc, actual_f1 = modeling.train_evaluate_pipeline(x_train, y_train, x_test, y_test, vectorizer, classifier)

        # Assert
        self.assertIsNotNone(actual_model)
        self.assertIsInstance(actual_acc, expected_metric_type)
        self.assertIsInstance(actual_f1, expected_metric_type)
        self.assertTrue(expected_min_value <= actual_acc <= expected_max_value)
        self.assertTrue(expected_min_value <= actual_f1 <= expected_max_value)
        

