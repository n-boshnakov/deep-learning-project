import unittest

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoModel

from fake_news_detector import modeling


class TestTrainEvaluatePipeline(unittest.TestCase):

    def test_when_text_and_labels_passed_then_returns_fitted_classifier_and_f1_and_precision_as_floats(
            self):
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

    def test_when_training_pytorch_model_then_returns_trained_module_and_f1_precision_and_history_dict(
            self):
        # Arrange
        x_train = torch.randn(10, 5)
        y_train = torch.randint(0, 2, (10, ))
        x_test = torch.randn(4, 5)
        y_test = torch.randint(0, 2, (4, ))

        train_loader = DataLoader(TensorDataset(x_train, y_train),
                                  batch_size=2)
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
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            device,
            epochs=1)

        # Assert
        self.assertIsInstance(trained_model, nn.Module)
        self.assertIsInstance(actual_f1, expected_metric_type)
        self.assertIsInstance(actual_precision, expected_metric_type)
        self.assertIsInstance(history, dict)

        self.assertTrue(expected_min_value <= actual_f1 <= expected_max_value)
        self.assertTrue(
            expected_min_value <= actual_precision <= expected_max_value)

    def test_when_training_hybrid_pytorch_model_then_supports_three_element_batches(
            self):
        # Arrange
        x_train_text = torch.randn(10, 5)
        x_train_meta = torch.randn(10, 3)
        y_train = torch.randint(0, 2, (10, ))

        x_test_text = torch.randn(4, 5)
        x_test_meta = torch.randn(4, 3)
        y_test = torch.randint(0, 2, (4, ))

        train_loader = DataLoader(TensorDataset(x_train_text, x_train_meta,
                                                y_train),
                                  batch_size=2)
        test_loader = DataLoader(TensorDataset(x_test_text, x_test_meta,
                                               y_test),
                                 batch_size=2)

        model = DummyHybridNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        device = torch.device("cpu")

        # Act
        trained_model, actual_f1, actual_precision, history = modeling.train_evaluate_pytorch_model(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            device,
            epochs=1)

        # Assert
        self.assertIsInstance(trained_model, nn.Module)
        self.assertIn("train_loss", history)
        self.assertIn("val_loss", history)


class TestBaselineEmbeddingNet(unittest.TestCase):

    def test_when_forward_pass_then_output_shape_is_batch_size_by_num_classes(
            self):
        # Arrange
        vocab_size = 100
        embed_dim = 10
        num_classes = 6
        batch_size = 4
        seq_len = 50

        model = modeling.BaselineEmbeddingNet(vocab_size, embed_dim,
                                              num_classes)
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Act
        output = model(dummy_input)

        # Assert
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (batch_size, num_classes))


class TestHybridRNNFakeNewsNet(unittest.TestCase):

    def test_when_forward_pass_with_different_rnns_then_output_shape_is_batch_size_by_num_classes(
            self):
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

                model = modeling.HybridRNNFakeNewsNet(vocab_size=vocab_size,
                                                      embed_dim=embed_dim,
                                                      hidden_dim=hidden_dim,
                                                      meta_dim=meta_dim,
                                                      num_classes=num_classes,
                                                      rnn_type=rnn_type)

                dummy_text_input = torch.randint(0, vocab_size,
                                                 (batch_size, seq_len))
                dummy_meta_input = torch.randn(batch_size, meta_dim)

                # Act
                output = model(dummy_text_input, dummy_meta_input)

                # Assert
                self.assertIsInstance(output, torch.Tensor)
                self.assertEqual(output.shape, (batch_size, num_classes))

    def test_when_invalid_rnn_type_passed_then_raises_value_error(self):
        # Arrange
        vocab_size = 100
        embed_dim = 10
        hidden_dim = 16
        meta_dim = 20
        num_classes = 6
        invalid_rnn_type = "INVALID_NETWORK"

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            modeling.HybridRNNFakeNewsNet(vocab_size=vocab_size,
                                          embed_dim=embed_dim,
                                          hidden_dim=hidden_dim,
                                          meta_dim=meta_dim,
                                          num_classes=num_classes,
                                          rnn_type=invalid_rnn_type)

        self.assertIn("rnn_type must be one of", str(context.exception))


class TestHybridModel(unittest.TestCase):

    def test_when_hybrid_bert_model_called_then_output_shape_is_batch_size_by_num_classes(
            self):
        # Arrange
        config = AutoConfig.from_pretrained('bert-base-uncased',
                                            num_hidden_layers=2,
                                            hidden_size=384,
                                            num_attention_heads=6)
        bert_model = AutoModel.from_config(config)

        meta_dim = 10
        num_classes = 6
        model = modeling.HybridBertFakeNewsNet('bert-base-uncased',
                                               meta_dim,
                                               num_classes=num_classes)

        model.bert = bert_model

        model.fc_out = nn.Linear(384 + 32, num_classes)

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len))
        meta_input = torch.randn(batch_size, meta_dim)

        # Act
        output = model(input_ids, attention_mask, meta_input)

        # Assert
        self.assertEqual(output.shape, (batch_size, num_classes))


class DummyHFOutput:
    """Mimics a HuggingFace model output with .loss and .logits."""

    def __init__(self, logits, labels):
        self.logits = logits
        self.loss = nn.CrossEntropyLoss()(logits, labels)


class DummyHFModel(nn.Module):
    """Mimics a text-only HuggingFace model (AutoModelForSequenceClassification)."""

    def __init__(self, num_classes=6):
        super().__init__()
        self.fc = nn.Linear(4, num_classes)

    def forward(self, input_ids, _attention_mask=None, labels=None, **_kwargs):
        logits = self.fc(input_ids.float())
        return DummyHFOutput(logits, labels)

    def parameters(self):
        return self.fc.parameters()


class DummyHybridTransformerModel(nn.Module):
    """Mimics a hybrid transformer model (HybridBertFakeNewsNet)."""

    def __init__(self, num_classes=6, meta_dim=3):
        super().__init__()
        self.fc = nn.Linear(4 + meta_dim, num_classes)

    def forward(self, input_ids, attention_mask=None, meta_input=None):
        combined = torch.cat([input_ids.float(), meta_input], dim=1)
        return self.fc(combined)


class TestTrainEvaluateTransformerModel(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.seq_len = 4
        self.meta_dim = 3
        self.num_classes = 6
        self.batch_size = 4

        input_ids = torch.randint(0, 10, (8, self.seq_len))
        attention_mask = torch.ones(8, self.seq_len, dtype=torch.long)
        labels = torch.randint(0, self.num_classes, (8, ))
        meta = torch.randn(8, self.meta_dim)

        from torch.utils.data import TensorDataset as TDS
        self.text_loader = DataLoader(TDS(input_ids, attention_mask, labels),
                                      batch_size=self.batch_size)
        self.hybrid_loader = DataLoader(TDS(input_ids, attention_mask, meta,
                                            labels),
                                        batch_size=self.batch_size)

    def test_when_training_text_only_transformer_then_returns_trained_module_and_f1_precision_and_history_dict(
            self):
        # Arrange
        model = DummyHFModel(num_classes=self.num_classes)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

        expected_metric_type = float
        expected_min_value = 0.0
        expected_max_value = 1.0

        # Act
        trained_model, actual_f1, actual_prec, history = modeling.train_evaluate_transformer_model(
            model,
            self.text_loader,
            self.text_loader,
            optimizer,
            self.device,
            epochs=1)

        # Assert
        self.assertIsInstance(trained_model, nn.Module)
        self.assertIsInstance(actual_f1, expected_metric_type)
        self.assertIsInstance(actual_prec, expected_metric_type)
        self.assertIsInstance(history, dict)
        self.assertIn("train_loss", history)
        self.assertIn("val_loss", history)
        self.assertIn("train_f1", history)
        self.assertIn("val_f1", history)
        self.assertTrue(expected_min_value <= actual_f1 <= expected_max_value)
        self.assertTrue(
            expected_min_value <= actual_prec <= expected_max_value)

    def test_when_training_hybrid_transformer_then_returns_trained_module_and_f1_precision_and_history_dict(
            self):
        # Arrange
        model = DummyHybridTransformerModel(num_classes=self.num_classes,
                                            meta_dim=self.meta_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

        expected_metric_type = float
        expected_min_value = 0.0
        expected_max_value = 1.0

        # Act
        trained_model, actual_f1, actual_prec, history = modeling.train_evaluate_transformer_model(
            model,
            self.hybrid_loader,
            self.hybrid_loader,
            optimizer,
            self.device,
            epochs=1,
            criterion=criterion)

        # Assert
        self.assertIsInstance(trained_model, nn.Module)
        self.assertIsInstance(actual_f1, expected_metric_type)
        self.assertIsInstance(actual_prec, expected_metric_type)
        self.assertTrue(expected_min_value <= actual_f1 <= expected_max_value)
        self.assertTrue(
            expected_min_value <= actual_prec <= expected_max_value)

    def test_when_training_transformer_then_history_contains_one_entry_per_epoch(
            self):
        # Arrange
        model = DummyHFModel(num_classes=self.num_classes)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        expected_epochs = 2

        # Act
        _, _, _, history = modeling.train_evaluate_transformer_model(
            model,
            self.text_loader,
            self.text_loader,
            optimizer,
            self.device,
            epochs=expected_epochs)

        # Assert
        self.assertEqual(len(history["train_loss"]), expected_epochs)
        self.assertEqual(len(history["val_f1"]), expected_epochs)

    def test_when_training_text_only_transformer_with_dict_batches_then_succeeds(
            self):
        # Arrange — dict loader without meta_features (lines 187-194, 249-256)
        input_ids = torch.randint(0, 10, (8, self.seq_len))
        attention_mask = torch.ones(8, self.seq_len, dtype=torch.long)
        labels = torch.randint(0, self.num_classes, (8, ))

        dataset = [{
            'input_ids': input_ids[i],
            'attention_mask': attention_mask[i],
            'labels': labels[i]
        } for i in range(8)]
        loader = DataLoader(dataset, batch_size=self.batch_size)

        model = DummyHFModel(num_classes=self.num_classes)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

        # Act
        trained_model, actual_f1, _, history = modeling.train_evaluate_transformer_model(
            model, loader, loader, optimizer, self.device, epochs=1)

        # Assert
        self.assertIsInstance(trained_model, nn.Module)
        self.assertIsInstance(actual_f1, float)
        self.assertIsInstance(history, dict)
        self.assertIn("train_loss", history)

    def test_when_training_hybrid_transformer_with_dict_batches_then_succeeds(
            self):
        # Arrange — dict loader with meta_features (lines 177-185, 239-247)
        input_ids = torch.randint(0, 10, (8, self.seq_len))
        attention_mask = torch.ones(8, self.seq_len, dtype=torch.long)
        labels = torch.randint(0, self.num_classes, (8, ))
        meta = torch.randn(8, self.meta_dim)

        dataset = [{
            'input_ids': input_ids[i],
            'attention_mask': attention_mask[i],
            'meta_features': meta[i],
            'labels': labels[i],
        } for i in range(8)]
        loader = DataLoader(dataset, batch_size=self.batch_size)

        model = DummyHybridTransformerModel(num_classes=self.num_classes,
                                            meta_dim=self.meta_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

        # Act
        trained_model, actual_f1, _, history = modeling.train_evaluate_transformer_model(
            model,
            loader,
            loader,
            optimizer,
            self.device,
            epochs=1,
            criterion=criterion)

        # Assert
        self.assertIsInstance(trained_model, nn.Module)
        self.assertIsInstance(actual_f1, float)
        self.assertIn("train_loss", history)
