import unittest
from unittest.mock import patch, mock_open
from data_loaders.metadata_loader import load_ocr_data

class TestMetadataLoader(unittest.TestCase):
    @patch('builtins.open', new_callable=mock_open, read_data='{"1": ["text"]}')
    @patch('os.path.exists', return_value=True)
    @patch('os.walk')
    def test_load_ocr_data(self, mock_walk, mock_exists, mock_file):
        image_paths = ['/path/to/image1.jpg']
        ocr_data, keywords = load_ocr_data(image_paths)
        self.assertIn('/path/to/image1.jpg', ocr_data)
        self.assertEqual(keywords, ['text'])

if __name__ == '__main__':
    unittest.main()
