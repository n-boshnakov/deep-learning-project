import os
import unittest
from unittest.mock import patch

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from fake_news_detector.utils import (plot_training_history, print_evaluation_metrics,
                                      save_model_pipeline)


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

class TestFileGenerationUtils(unittest.TestCase):

    def setUp(self) -> None:
        # Define the paths as class attributes so they can be accessed anywhere
        self.base_dir = "test_models_dir"
        self.file_name = "test_pipeline.pkl"
        self.expected_path = os.path.join(self.base_dir, self.file_name)

    def tearDown(self) -> None:
        # Clean up the file first, then the directory
        if os.path.exists(self.expected_path):
            os.remove(self.expected_path)
        if os.path.exists(self.base_dir):
            os.rmdir(self.base_dir)

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

class TestPlotTrainingHistoryUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.base_dir = "test_plots_dir"
        self.experiment_name = "Test Experiment"
        self.expected_path = os.path.join(self.base_dir, "Test_Experiment_history.png")

    def tearDown(self) -> None:
        if os.path.exists(self.expected_path):
            os.remove(self.expected_path)
        if os.path.exists(self.base_dir):
            os.rmdir(self.base_dir)

    def test_when_valid_history_passed_then_saves_plot_successfully(self):
        # Arrange
        dummy_history = {
            "train_loss": [0.8, 0.5],
            "val_loss": [0.9, 0.6],
            "train_f1": [0.2, 0.4],
            "val_f1": [0.1, 0.3]
        }

        # Act
        plot_training_history(dummy_history, self.experiment_name, self.base_dir)

        # Assert
        self.assertTrue(os.path.exists(self.expected_path))