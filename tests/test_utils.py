import unittest
import os
import joblib

from unittest.mock import patch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from fake_news_detector.utils import print_evaluation_metrics
from fake_news_detector.utils import save_model_pipeline

class TestUtils(unittest.TestCase):

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