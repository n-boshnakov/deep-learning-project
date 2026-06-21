import os
import unittest

import pandas as pd
import torch

from fake_news_detector import parse_data


class TestParseData(unittest.TestCase):

    def setUp(self) -> None:
        self.test_filepath = "dummy.tsv"

    def tearDown(self) -> None:
        if os.path.exists(self.test_filepath):
            os.remove(self.test_filepath)

    def test_when_loading_and_splitting_data_then_returns_only_rows_with_no_missing_crucial_information(
            self):
        # Arrange
        expected_valid_rows = 3

        with open(self.test_filepath, "w", encoding="utf-8") as f:
            f.write(
                "1\ttrue\tTest sentence 1.\tlabel-1,label-2,label-3\tivan-andreev\tworker\tBulgaria\tright\t0\t1\t1\t1\t1\tIn a news story\n"
            )
            f.write(
                "2\tfalse\tTest sentence 2, that is false.\tlabel-6\tmaria-krumova\tsecretary\tRomania\tleft\t2\t1\t1\t1\t1\tAn email\n"
            )
            f.write(
                "3\ttrue\tTest sentence 3. This one is also true.\tlabel-1, label-4\tignat-ivanov\tit-specialist\tBulgaria\tcenter\t0\t1\t1\t1\t1\tPress conference\n"
            )
            f.write(
                "4\t\tTest sentence 4, which is missing a truth class.\tlabel-3, label-4\tmaria-krumova\tsecretary\tRomania\tleft\t2\t1\t1\t1\t1\tAn email\n"
            )

        # Act
        actual_x, actual_y = parse_data.load_and_split_data(self.test_filepath)

        # Assert
        self.assertEqual(expected_valid_rows, len(actual_x))
        self.assertEqual(expected_valid_rows, len(actual_y))


class TestParseDataDeepLearning(unittest.TestCase):

    def test_when_building_vocab_then_returns_correct_dict_with_special_tokens(
            self):
        # Arrange
        texts = pd.Series(["hello world", "hello test"])
        expected_pad_idx = 0
        expected_unk_idx = 1

        # Act
        vocab = parse_data.build_vocab(texts, max_vocab_size=10)

        # Assert
        self.assertIsInstance(vocab, dict)
        self.assertIn("<PAD>", vocab)
        self.assertIn("<UNK>", vocab)
        self.assertEqual(vocab["<PAD>"], expected_pad_idx)
        self.assertEqual(vocab["<UNK>"], expected_unk_idx)
        self.assertIn("hello", vocab)

    def test_when_text_to_indices_called_then_pads_and_truncates_correctly(
            self):
        # Arrange
        texts = pd.Series(["hello world", "a very long sentence to truncate"])
        vocab = {"<PAD>": 0, "<UNK>": 1, "hello": 2, "world": 3, "a": 4}
        expected_shape = (2, 4)
        expected_padded = [2, 3, 0, 0]
        expected_truncated = [4, 1, 1, 1]

        # Act
        actual_tensor = parse_data.text_to_indices(texts, vocab, max_seq_len=4)

        # Assert
        self.assertIsInstance(actual_tensor, torch.Tensor)
        self.assertEqual(actual_tensor.shape, expected_shape)
        self.assertEqual(actual_tensor[0].tolist(), expected_padded)
        self.assertEqual(actual_tensor[1].tolist(), expected_truncated)

    def test_when_liardataset_instantiated_then_returns_valid_tensors(self):
        # Arrange
        x = pd.Series(["fake news", "real news"])
        y = pd.Series(["pants-fire", "true"])
        vocab = {"<PAD>": 0, "<UNK>": 1, "fake": 2, "news": 3, "real": 4}
        expected_length = 2
        expected_first_label = 0

        # Act
        dataset = parse_data.LiarDataset(x, y, vocab, max_seq_len=3)
        features, label = dataset[0]

        # Assert
        self.assertEqual(len(dataset), expected_length)
        self.assertIsInstance(features, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
        self.assertEqual(label.item(), expected_first_label)
