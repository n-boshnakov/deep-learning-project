import unittest
import numpy as np
from src import features

class TestVectorizers(unittest.TestCase):

    def test_when_text_is_passed_to_spacy_then_returns_matrix(self):
        # Arrange
        dummy_txt = ["Fake text", "this is some more text", "Another statement from a political leader."]
        vectorizer = features.SpacyVectorizer(model_name="en_core_web_md")
        expected_matrix_rows = 3
        # 300 is the standard for the en_core_web spaCy model
        expected_matrix_columns = 300

        # Act
        actual_matrix = vectorizer.fit_transform(dummy_txt)

        # Assert
        self.assertIsInstance(actual_matrix, np.ndarray)
        self.assertEqual(expected_matrix_rows, actual_matrix.shape[0])
        self.assertEqual(expected_matrix_columns, actual_matrix.shape[1])

    def test_when_text_is_passed_to_gensim_then_returns_matrix(self):
        # Arrange
        dummy_txt = ["Fake text", "this is some more text", "Another statement from a political leader."]
        vectorizer = features.GensimVectorizer(model_name="glove-twitter-25")
        expected_matrix_rows = 3
        # 25 is the standard for the glove-twitter-25 GenSim model
        expected_matrix_columns = 25

        # Act
        actual_matrix = vectorizer.fit_transform(dummy_txt)

        # Assert
        self.assertIsInstance(actual_matrix, np.ndarray)
        self.assertEqual(expected_matrix_rows, actual_matrix.shape[0])
        self.assertEqual(expected_matrix_columns, actual_matrix.shape[1])

