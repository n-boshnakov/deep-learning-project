import unittest

import numpy as np

from fake_news_detector import features


class TestVectorizers(unittest.TestCase):

    def test_when_text_is_passed_to_spacy_transform_then_returns_matrix(self):
        # Arrange
        dummy_txt = [
            "Fake text", "this is some more text",
            "Another statement from a political leader."
        ]
        vectorizer = features.SpacyVectorizer("en_core_web_md")
        expected_matrix_rows = 3
        # 300 is the standard for the en_core_web spaCy model
        expected_matrix_columns = 300

        # Act
        actual_matrix = vectorizer.fit_transform(dummy_txt)

        # Assert
        self.assertIsInstance(actual_matrix, np.ndarray)
        self.assertEqual(expected_matrix_rows, actual_matrix.shape[0])
        self.assertEqual(expected_matrix_columns, actual_matrix.shape[1])

    def test_when_text_is_passed_to_spacy_fit_then_returns_self(self):
        # Arrange
        vectorizer = features.SpacyVectorizer("en_core_web_md")
        dummy_data = ["test sentence"]
        expected_result = vectorizer

        # Act
        actual_result = vectorizer.fit(dummy_data)

        # Assert
        self.assertEqual(expected_result, actual_result)

    def test_when_spacy_fit_transform_called_then_returns_numpy_array_with_one_row_per_input(self):
        # Arrange
        vectorizer = features.SpacyVectorizer("en_core_web_md")
        dummy_data = ["test sentence"]

        # Act
        result = vectorizer.fit_transform(dummy_data)

        # Assert
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 1)

    def test_when_text_is_passed_to_gensim_transform_then_returns_matrix(self):
        # Arrange
        dummy_txt = [
            "Fake text", "this is some more text",
            "Another statement from a political leader."
        ]
        vectorizer = features.GensimVectorizer("glove-twitter-25")
        expected_matrix_rows = 3
        # 25 is the standard for the glove-twitter-25 GenSim model
        expected_matrix_columns = 25

        # Act
        actual_matrix = vectorizer.fit_transform(dummy_txt)

        # Assert
        self.assertIsInstance(actual_matrix, np.ndarray)
        self.assertEqual(expected_matrix_rows, actual_matrix.shape[0])
        self.assertEqual(expected_matrix_columns, actual_matrix.shape[1])

    def test_when_text_is_passed_to_gensim_fit_then_returns_self(self):
        # Arrange
        vectorizer = features.GensimVectorizer("glove-twitter-25")
        dummy_data = ["test sentence"]
        expected_result = vectorizer

        # Act
        actual_result = vectorizer.fit(dummy_data)

        # Assert
        self.assertEqual(expected_result, actual_result)

    def test_when_gensim_fit_transform_called_then_returns_numpy_array_with_one_row_per_input(self):
        # Arrange
        vectorizer = features.GensimVectorizer("glove-twitter-25")
        dummy_data = ["test sentence", ""]

        # Act
        result = vectorizer.fit_transform(dummy_data)

        # Assert
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 2)
