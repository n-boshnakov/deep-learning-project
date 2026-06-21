import unittest

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

from fake_news_detector import modeling


class TestTrainEvaluatePipeline(unittest.TestCase):

    def test_when_valid_data_passed_then_returns_model_and_float_metrics(self):
        # Arrange
        x_train = [
            "this is test data", "this text contains only numbers",
            "this is real data", "this statement is true"
        ]
        y_train = ["true", "false", "false", "true"]

        x_test = ["this statement is false", "123456"]
        y_test = ["false", "true"]

        vectorizer = TfidfVectorizer()
        classifier = LogisticRegression()

        expected_metric_type = float
        expected_min_value = 0.0
        expected_max_value = 1.0

        # Act
        actual_model, actual_acc, actual_f1 = modeling.train_evaluate_pipeline(
            x_train, y_train, x_test, y_test, vectorizer, classifier)

        # Assert
        self.assertIsNotNone(actual_model)
        self.assertIsInstance(actual_acc, expected_metric_type)
        self.assertIsInstance(actual_f1, expected_metric_type)
        self.assertTrue(expected_min_value <= actual_acc <= expected_max_value)
        self.assertTrue(expected_min_value <= actual_f1 <= expected_max_value)


class TestTrainEvaluatePytorchModel(unittest.TestCase):

    def test_when_training_pytorch_model_then_returns_model_and_valid_metrics(self):
        # Arrange
        x_train = torch.randn(10, 5)
        y_train = torch.randint(0, 2, (10,))
        x_test = torch.randn(4, 5)
        y_test = torch.randint(0, 2, (4,))

        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=2)
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=2)

        model = nn.Linear(5, 2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        device = torch.device("cpu")

        expected_metric_type = float
        expected_min_value = 0.0
        expected_max_value = 1.0

        # Act
        trained_model, actual_f1, actual_precision, history = modeling.train_evaluate_pytorch_model(
            model, train_loader, test_loader, criterion, optimizer, device, epochs=1
        )

        # Assert
        self.assertIsInstance(trained_model, nn.Module)
        self.assertIsInstance(actual_f1, expected_metric_type)
        self.assertIsInstance(actual_precision, expected_metric_type)
        self.assertIsInstance(history, dict)
        
        self.assertTrue(expected_min_value <= actual_f1 <= expected_max_value)
        self.assertTrue(expected_min_value <= actual_precision <= expected_max_value)

        self.assertIn("train_loss", history)
        self.assertIn("val_loss", history)
        self.assertIn("train_f1", history)
        self.assertIn("val_f1", history)
        self.assertEqual(len(history["train_loss"]), 1)