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


# Dummy model to test the training loop with 3 elements (text, meta, label)
class DummyHybridNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 5 text features + 3 meta features = 8 total input features
        self.fc = nn.Linear(8, 2) 

    def forward(self, text, meta):
        combined = torch.cat((text, meta), dim=1)
        return self.fc(combined)


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

    def test_when_training_hybrid_pytorch_model_then_supports_three_element_batches(self):
        # Arrange
        x_train_text = torch.randn(10, 5)
        x_train_meta = torch.randn(10, 3)
        y_train = torch.randint(0, 2, (10,))
        
        x_test_text = torch.randn(4, 5)
        x_test_meta = torch.randn(4, 3)
        y_test = torch.randint(0, 2, (4,))

        train_loader = DataLoader(TensorDataset(x_train_text, x_train_meta, y_train), batch_size=2)
        test_loader = DataLoader(TensorDataset(x_test_text, x_test_meta, y_test), batch_size=2)

        model = DummyHybridNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        device = torch.device("cpu")

        # Act
        trained_model, actual_f1, actual_precision, history = modeling.train_evaluate_pytorch_model(
            model, train_loader, test_loader, criterion, optimizer, device, epochs=1
        )

        # Assert
        self.assertIsInstance(trained_model, nn.Module)
        self.assertIn("train_loss", history)
        self.assertIn("val_loss", history)


class TestBaselineEmbeddingNet(unittest.TestCase):

    def test_when_forward_pass_then_returns_correct_shape(self):
        # Arrange
        vocab_size = 100
        embed_dim = 10
        num_classes = 6
        batch_size = 4
        seq_len = 50

        model = modeling.BaselineEmbeddingNet(vocab_size, embed_dim, num_classes)
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Act
        output = model(dummy_input)

        # Assert
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (batch_size, num_classes))


class TestHybridRNNFakeNewsNet(unittest.TestCase):

    def test_when_forward_pass_with_different_rnns_then_returns_correct_shape(self):
        # Arrange
        vocab_size = 100
        embed_dim = 10
        hidden_dim = 16
        meta_dim = 20
        num_classes = 6
        batch_size = 4
        seq_len = 50

        rnn_types_to_test = ['GRU', 'LSTM', 'RNN']

        for rnn_type in rnn_types_to_test:
            # Use subTest so if one fails, the others still run
            with self.subTest(rnn_type=rnn_type):
                
                model = modeling.HybridRNNFakeNewsNet(
                    vocab_size=vocab_size, 
                    embed_dim=embed_dim, 
                    hidden_dim=hidden_dim, 
                    meta_dim=meta_dim, 
                    num_classes=num_classes,
                    rnn_type=rnn_type
                )
                
                dummy_text_input = torch.randint(0, vocab_size, (batch_size, seq_len))
                dummy_meta_input = torch.randn(batch_size, meta_dim)

                # Act
                output = model(dummy_text_input, dummy_meta_input)

                # Assert
                self.assertIsInstance(output, torch.Tensor)
                self.assertEqual(output.shape, (batch_size, num_classes))