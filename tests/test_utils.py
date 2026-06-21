import os
import pickle
import unittest
from unittest.mock import patch

import joblib
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from fake_news_detector.utils import (plot_training_history, print_evaluation_metrics,
                                      save_model_pipeline, save_pytorch_model)


class TestUtils(unittest.TestCase):

    @patch('builtins.print')
    def test_when_print_evaluation_metrics_called_then_prints_correctly(self, mock_print):
        # Arrange
        experiment_name = "H01 - baseline"
        f1_score = 0.1835
        precision_score = 0.2229

        # Act
        print_evaluation_metrics(experiment_name, f1_score, precision_score)

        # Assert
        self.assertEqual(mock_print.call_count, 3)
        
        first_print_call_args = mock_print.call_args_list[0][0][0]
        self.assertIn("H01 - baseline", first_print_call_args)

class BaseFileTest(unittest.TestCase):
    """
    Базов клас, който автоматично създава временна папка преди тест
    и я изчиства напълно (заедно с файловете в нея) след теста.
    """
    def setUp(self) -> None:
        self.base_dir = "test_temp_dir"
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def tearDown(self) -> None:
        if os.path.exists(self.base_dir):
            for file in os.listdir(self.base_dir):
                os.remove(os.path.join(self.base_dir, file))
            os.rmdir(self.base_dir)

class TestSaveModelPipeline(BaseFileTest):

    def setUp(self) -> None:
        super().setUp() # Извиква логиката от базовия клас за създаване на папката
        self.file_name = "test_pipeline.pkl"
        self.expected_path = os.path.join(self.base_dir, self.file_name)

    def test_when_saving_pipeline_then_pkl_file_is_created_and_contains_models(self):
        # Arrange
        vectorizer = TfidfVectorizer()
        classifier = LogisticRegression()

        # Act
        save_model_pipeline(vectorizer, classifier, self.file_name, self.base_dir)

        # Assert
        self.assertTrue(os.path.exists(self.expected_path))
        loaded_data = joblib.load(self.expected_path)
        self.assertIn("vectorizer", loaded_data)
        self.assertIn("classifier", loaded_data)


class TestPlotTrainingHistoryUtils(BaseFileTest):

    def setUp(self) -> None:
        super().setUp()
        self.experiment_name = "Test Experiment"
        self.expected_path = os.path.join(self.base_dir, "Test_Experiment_history.png")

    def test_when_valid_history_passed_then_saves_plot_successfully(self):
        # Arrange
        dummy_history = {
            "train_loss": [0.8, 0.5], "val_loss": [0.9, 0.6],
            "train_f1": [0.2, 0.4], "val_f1": [0.1, 0.3]
        }

        # Act
        plot_training_history(dummy_history, self.experiment_name, self.base_dir)

        # Assert
        self.assertTrue(os.path.exists(self.expected_path))


class DummyNet(nn.Module):
    def __init__(self):
        super(DummyNet, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class TestSavePytorchModel(BaseFileTest):

    def setUp(self) -> None:
        super().setUp()
        # Arrange
        self.dummy_model = DummyNet()
        self.dummy_vocab = {"<PAD>": 0, "<UNK>": 1, "test": 2}
        
        self.model_name = "test_weights.pth"
        self.vocab_name = "test_vocab.pkl"
        
        self.expected_model_path = os.path.join(self.base_dir, self.model_name)
        self.expected_vocab_path = os.path.join(self.base_dir, self.vocab_name)

    def test_when_save_pytorch_model_called_then_creates_files(self):
        # Act
        save_pytorch_model(
            model=self.dummy_model,
            word2idx=self.dummy_vocab,
            model_name=self.model_name,
            vocab_name=self.vocab_name,
            base_dir=self.base_dir
        )

        # Assert
        self.assertTrue(os.path.exists(self.expected_model_path))
        self.assertTrue(os.path.exists(self.expected_vocab_path))

        with open(self.expected_vocab_path, 'rb') as f:
            loaded_vocab = pickle.load(f)
        self.assertEqual(loaded_vocab, self.dummy_vocab)