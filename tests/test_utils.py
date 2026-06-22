import os
import unittest
from unittest.mock import patch

import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from fake_news_detector.utils import (plot_training_history, print_evaluation_metrics,
                                      save_artifacts)


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

class TestSaveArtifacts(BaseFileTest):

    def setUp(self) -> None:
        super().setUp()
        self.model_filename = "test_weights.pth"
        self.vocab_filename = "test_vocab.pkl"
        self.prep_filename = "test_prep.pkl"

    def test_when_saving_various_artifacts_then_files_are_created_correctly(self):
        dummy_model = nn.Linear(10, 2)
        dummy_vocab = {"<PAD>": 0, "test": 1}
        dummy_preprocessor = LogisticRegression()
        
        artifacts = {
            self.model_filename: dummy_model,
            self.vocab_filename: dummy_vocab,
            self.prep_filename: dummy_preprocessor
        }

        # Act
        save_artifacts(artifacts, base_dir=self.base_dir)

        # Assert
        expected_model_path = os.path.join(self.base_dir, self.model_filename)
        expected_vocab_path = os.path.join(self.base_dir, self.vocab_filename)
        expected_prep_path = os.path.join(self.base_dir, self.prep_filename)

        self.assertTrue(os.path.exists(expected_model_path))
        self.assertTrue(os.path.exists(expected_vocab_path))
        self.assertTrue(os.path.exists(expected_prep_path))