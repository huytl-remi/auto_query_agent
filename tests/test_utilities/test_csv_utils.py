import unittest
from utilities.csv_utils import calculate_frame_idx

class TestCSVUtils(unittest.TestCase):
    def test_calculate_frame_idx(self):
        time_str = '1:30'
        fps = 30
        frame_idx = calculate_frame_idx(time_str, fps)
        expected_frame_idx = 1 * 60 * 30 + 30 * 30  # 90 seconds * 30 fps
        self.assertEqual(frame_idx, expected_frame_idx)

    def test_calculate_frame_idx_invalid_time(self):
        time_str = 'invalid'
        fps = 30
        with self.assertRaises(ValueError):
            calculate_frame_idx(time_str, fps)

if __name__ == '__main__':
    unittest.main()
