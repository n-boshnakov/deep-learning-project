import os
import unittest

from fake_news_detector import parse_data


class TestParseData(unittest.TestCase):

    def test_when_loading_and_splitting_data_then_returns_only_rows_with_no_missing_crucial_information(self):
        # Arrange
        test_filepath = "dummy.tsv"
        expected_valid_rows = 3
        
        with open(test_filepath, "w", encoding="utf-8") as f:
            f.write("1\ttrue\tTest sentence 1.\tlabel-1,label-2,label-3\tivan-andreev\tworker\tBulgaria\tright\t0\t1\t1\t1\t1\tIn a news story\n")
            f.write("2\tfalse\tTest sentence 2, that is false.\tlabel-6\tmaria-krumova\tsecretary\tRomania\tleft\t2\t1\t1\t1\t1\tAn email\n")
            f.write("3\ttrue\tTest sentence 3. This one is also true.\tlabel-1, label-4\tignat-ivanov\tit-specialist\tBulgaria\tcenter\t0\t1\t1\t1\t1\tPress conference\n")
            f.write("4\t\tTest sentence 4, which is missing a truth class.\tlabel-3, label-4\tmaria-krumova\tsecretary\tRomania\tleft\t2\t1\t1\t1\t1\tAn email\n")

        # Act
        actual_x, actual_y = parse_data.load_and_split_data(test_filepath)

        # Assert
        self.assertEqual(expected_valid_rows, len(actual_x))
        self.assertEqual(expected_valid_rows, len(actual_y))
        
        os.remove(test_filepath)