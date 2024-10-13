# tests/test_services/test_search_service.py
import unittest
from unittest.mock import MagicMock
from services.search_service import perform_search

class TestSearchService(unittest.TestCase):
    def setUp(self):
        # Mock model, index, and id2img_fps
        self.mock_model = MagicMock()
        self.mock_index = MagicMock()
        self.id2img_fps = {
            '0': {'image_path': 'image0.jpg'},
            '1': {'image_path': 'image1.jpg'},
            '2': {'image_path': 'image2.jpg'},
        }

    def test_perform_search_clip(self):
        # Setup
        def mock_search_image_by_text(model, index, text_query, top_k):
            return ([0, 1], [0.1, 0.2])

        with unittest.mock.patch('utilities.model_utils.search_image_by_text', side_effect=mock_search_image_by_text):
            search_method = "CLIP"
            text_query = "test query"
            top_k = 2
            deleted_images = {'image1.jpg'}

            # Execute
            image_paths = perform_search(
                search_method=search_method,
                model=self.mock_model,
                index=self.mock_index,
                text_query=text_query,
                top_k=top_k,
                deleted_images=deleted_images,
                id2img_fps=self.id2img_fps
            )

            # Verify
            self.assertEqual(image_paths, ['image0.jpg', 'image2.jpg'])

    def test_perform_search_captioning(self):
        # Mock search_image_by_text_with_captioning
        with unittest.mock.patch('utilities.model_utils.search_image_by_text_with_captioning', return_value=['image1.jpg', 'image2.jpg']):
            search_method = "Captioning"
            text_query = "caption query"
            top_k = 2
            deleted_images = {'image2.jpg'}

            # Execute
            image_paths = perform_search(
                search_method=search_method,
                model=self.mock_model,
                index=self.mock_index,
                text_query=text_query,
                top_k=top_k,
                deleted_images=deleted_images,
                id2img_fps=self.id2img_fps
            )

            # Verify
            self.assertEqual(image_paths, ['image1.jpg'])

    def test_perform_search_ocr(self):
        # Mock search_images_by_ocr
        with unittest.mock.patch('utilities.model_utils.search_images_by_ocr', return_value=['image0.jpg', 'image2.jpg']):
            search_method = "OCR"
            text_query = "ocr query"
            top_k = 2
            deleted_images = {'image0.jpg'}

            # Execute
            image_paths = perform_search(
                search_method=search_method,
                model=self.mock_model,
                index=self.mock_index,
                text_query=text_query,
                top_k=top_k,
                deleted_images=deleted_images,
                id2img_fps=self.id2img_fps
            )

            # Verify
            self.assertEqual(image_paths, ['image2.jpg'])

    def test_perform_search_invalid_method(self):
        with self.assertRaises(ValueError):
            perform_search(
                search_method="InvalidMethod",
                model=self.mock_model,
                index=self.mock_index,
                text_query="test",
                top_k=1,
                deleted_images=set(),
                id2img_fps=self.id2img_fps
            )

if __name__ == '__main__':
    unittest.main()
