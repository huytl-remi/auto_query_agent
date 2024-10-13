# tests/test_utilities/test_video_utils.py
import unittest
from unittest.mock import patch, MagicMock
from utilities.csv_utils import calculate_frame_idx

class TestVideoUtils(unittest.TestCase):
    def test_calculate_frame_idx_valid(self):
        time_str = '2:30'
        fps = 30
        frame_idx = calculate_frame_idx(time_str, fps)
        expected = (2 * 60 + 30) * fps  # 150 * 30 = 4500
        self.assertEqual(frame_idx, expected)

    def test_calculate_frame_idx_invalid_format(self):
        time_str = 'invalid'
        fps = 30
        with self.assertRaises(ValueError):
            calculate_frame_idx(time_str, fps)

    def test_calculate_frame_idx_non_integer(self):
        time_str = '1:30.5'
        fps = 30
        with self.assertRaises(ValueError):
            calculate_frame_idx(time_str, fps)

if __name__ == '__main__':
    unittest.main()
