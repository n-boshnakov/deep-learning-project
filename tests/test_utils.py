import unittest
from unittest.mock import patch

from fake_news_detector.utils import print_evaluation_metrics


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