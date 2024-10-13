import unittest
from unittest.mock import patch, MagicMock
from services.search_service import perform_search
import numpy as np

class TestApp(unittest.TestCase):
    @patch('utilities.model_utils.faiss')
    def test_perform_search_clip(self, mock_faiss):
        # Mock the FAISS index
        index = MagicMock()
        mock_faiss.read_index.return_value = index

        # Mock index.search to return expected values
        mock_D = np.array([[0.1, 0.2]])  # Distances array
        mock_I = np.array([[0, 1]])      # Indices array
        index.search.return_value = (mock_D, mock_I)

        # Mock encode_text to return a tensor-like object with cpu().numpy() chain
        mock_text_features = MagicMock()
        mock_text_features.cpu.return_value.numpy.return_value = np.array([[0.5, 0.5]])

        with patch('utilities.model_utils.encode_text', return_value=mock_text_features):
            # Test parameters
            model = MagicMock()
            text_query = 'test query'
            top_k = 2
            deleted_images = set()  # No deleted images
            id2img_fps = {
                '0': {'image_path': 'image1.jpg'},
                '1': {'image_path': 'image2.jpg'},
            }

            # Call the perform_search function
            image_paths = perform_search(
                'CLIP',
                model,
                index,
                text_query,
                top_k,
                deleted_images,
                id2img_fps
            )

            # Assert that the image paths returned are correct
            self.assertEqual(image_paths, ['image1.jpg', 'image2.jpg'])

if __name__ == '__main__':
    unittest.main()
