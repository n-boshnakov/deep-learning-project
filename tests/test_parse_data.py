import os
import unittest

import pandas as pd
import torch

from fake_news_detector import parse_data


class TestParseData(unittest.TestCase):

    def setUp(self) -> None:
        self.test_filepath = "dummy.tsv"

        with open(self.test_filepath, "w", encoding="utf-8") as f:
            # 1. Valid row
            f.write(
                "1\ttrue\tTest sentence 1.\tlabel-1\tivan\tworker\tBulgaria\tright\t0\t1\t1\t1\t1\tnews\n"
            )
            # 2. Valid row
            f.write(
                "2\tfalse\tTest sentence 2.\tlabel-6\tmaria\tsecretary\tRomania\tleft\t2\t1\t1\t1\t1\temail\n"
            )
            # 3. Valid row
            f.write(
                "3\ttrue\tTest sentence 3.\tlabel-1\tignat\tit-spec\tBulgaria\tcenter\t0\t1\t1\t1\t1\tpress\n"
            )
            # 4. Missing label (Should be dropped)
            f.write(
                "4\t\tTest sentence 4.\tlabel-3\tmaria\tsecretary\tRomania\tleft\t2\t1\t1\t1\t1\temail\n"
            )
            # 5. Missing statement (Should be dropped)
            f.write(
                "5\tfalse\t\tlabel-3\tmaria\tsecretary\tRomania\tleft\t2\t1\t1\t1\t1\temail\n"
            )
            # 6. Valid statement and label, but missing ALL metadata (Should be filled)
            f.write("6\thalf-true\tTest sentence 6.\t\t\t\t\t\t\t\t\t\t\t\n")

    def tearDown(self) -> None:
        if os.path.exists(self.test_filepath):
            os.remove(self.test_filepath)

    def test_when_loading_and_splitting_data_then_returns_only_rows_with_no_missing_crucial_information(
            self):
        # Arrange
        expected_valid_rows = 4  # Rows 1, 2, 3, 6 are valid for this function

        # Act
        actual_x, actual_y = parse_data.load_and_split_data(self.test_filepath)

        # Assert
        self.assertEqual(expected_valid_rows, len(actual_x))
        self.assertEqual(expected_valid_rows, len(actual_y))

    def test_when_loading_hybrid_data_then_fills_nans_and_drops_invalid(self):
        # Arrange
        expected_valid_rows = 4  # Rows 1, 2, 3, 6

        # Act
        df = parse_data.load_hybrid_data(self.test_filepath)

        # Assert
        self.assertEqual(len(df), expected_valid_rows)

        # Verify that row 6 (which is at index 3 in the dataframe) got its NaNs filled
        row_6 = df.iloc[3]

        # Check numerical fills
        self.assertEqual(row_6["barely_true_counts"], 0.0)
        self.assertEqual(row_6["false_counts"], 0.0)

        # Check categorical fills
        self.assertEqual(row_6["speaker"], "unknown")
        self.assertEqual(row_6["party_affiliation"], "unknown")


class TestParseDataDeepLearning(unittest.TestCase):

    def test_when_building_vocab_then_returns_dict_with_pad_at_index_zero_and_unk_at_index_one(
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

    def test_when_text_to_indices_called_then_short_sequences_are_padded_and_long_sequences_are_truncated(
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

    def test_when_liardataset_instantiated_then_returns_long_tensor_for_features_and_label_tensor_per_item(
            self):
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


class TestHybridDataDeepLearning(unittest.TestCase):

    def setUp(self) -> None:
        data = {
            "statement": ["fake text", "real text"],
            "label": ["false", "true"],
            "barely_true_counts": [1.0, 0.0],
            "false_counts": [2.0, 0.0],
            "half_true_counts": [0.0, 0.0],
            "mostly_true_counts": [0.0, 1.0],
            "pants_on_fire_counts": [1.0, 0.0],
            "party_affiliation": ["republican", "democrat"],
            "state_info": ["Texas", "California"],
            "speaker_job": ["governor", "senator"],
            "speaker": ["smith", "jones"],
            "context": ["tv", "radio"],
            "subjects": ["tax", "health"]
        }
        self.dummy_df = pd.DataFrame(data)
        self.dummy_vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
            "fake": 2,
            "text": 3,
            "real": 4
        }

    def test_when_preprocessing_metadata_then_returns_float32_tensor_with_more_columns_than_numeric_inputs(
            self):
        # Arrange
        preprocessor = parse_data.MetadataPreprocessor()

        # Act
        preprocessor.fit(self.dummy_df)
        tensor_out = preprocessor.transform(self.dummy_df)

        # Assert
        self.assertIsInstance(tensor_out, torch.Tensor)
        self.assertEqual(tensor_out.dtype, torch.float32)
        self.assertEqual(tensor_out.shape[0], 2)
        self.assertTrue(tensor_out.shape[1] > 5)

    def test_when_using_hybrid_dataset_then_returns_three_elements(self):
        # Arrange
        preprocessor = parse_data.MetadataPreprocessor()
        preprocessor.fit(self.dummy_df)
        meta_tensor = preprocessor.transform(self.dummy_df)
        expected_length = 2

        # Act
        dataset = parse_data.LiarHybridDataset(self.dummy_df,
                                               meta_tensor,
                                               self.dummy_vocab,
                                               max_seq_len=5)

        # Assert
        self.assertEqual(len(dataset), expected_length)

        x_text, x_meta, y = dataset[0]

        self.assertIsInstance(x_text, torch.Tensor)
        self.assertIsInstance(x_meta, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)

        self.assertEqual(y.item(), 1)


class MockTokenizer:
    # A simple mock tokenizer to avoid downloading HuggingFace models during unit tests.
    def __call__(self,
                 text,
                 add_special_tokens=True,
                 max_length=50,
                 padding='max_length',
                 truncation=True,
                 return_attention_mask=True,
                 return_tensors='pt'):
        return {
            'input_ids': torch.ones((1, max_length), dtype=torch.long),
            'attention_mask': torch.ones((1, max_length), dtype=torch.long)
        }


class TestTransformerDatasets(unittest.TestCase):

    def setUp(self) -> None:
        self.texts = ["fake statement", "true statement", "another one"]
        self.labels = [0, 5, 2]
        self.meta_features = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            dtype=torch.float32)
        self.tokenizer = MockTokenizer()
        self.max_len = 8

    def test_when_transformer_dataset_instantiated_then_returns_dict_with_input_ids_attention_mask_and_labels(
            self):
        # Arrange
        dataset = parse_data.LiarTransformerDataset(self.texts, self.labels,
                                                    self.tokenizer,
                                                    self.max_len)

        # Act
        length = len(dataset)
        item = dataset[0]

        # Assert
        self.assertEqual(length, 3)
        self.assertIsInstance(item, dict)
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('labels', item)

        self.assertEqual(item['input_ids'].shape, torch.Size([self.max_len]))
        self.assertEqual(item['attention_mask'].shape,
                         torch.Size([self.max_len]))
        self.assertIsInstance(item['labels'], torch.Tensor)
        self.assertEqual(item['labels'].item(), 0)

    def test_when_hybrid_transformer_dataset_instantiated_then_returns_meta_features(
            self):
        # Arrange
        dataset = parse_data.LiarHybridBertDataset(self.texts,
                                                   self.meta_features,
                                                   self.labels, self.tokenizer,
                                                   self.max_len)

        # Act
        length = len(dataset)
        item = dataset[1]

        # Assert
        self.assertEqual(length, 3)
        self.assertIsInstance(item, dict)
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('meta_features', item)
        self.assertIn('labels', item)

        # Verify meta features mapping (should match the second row of the setup tensor)
        self.assertTrue(
            torch.equal(item['meta_features'], self.meta_features[1]))
        self.assertEqual(item['labels'].item(), 5)
