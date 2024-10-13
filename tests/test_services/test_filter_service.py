# tests/test_services/test_filter_service.py
import unittest
from unittest.mock import patch, MagicMock
from services.filter_service import apply_metadata_filters

class TestFilterService(unittest.TestCase):
    @patch('data_loaders.metadata_loader.load_ocr_data')
    @patch('data_loaders.metadata_loader.load_object_data')
    @patch('data_loaders.metadata_loader.load_object_count_data')
    def test_apply_metadata_filters(self, mock_load_counts, mock_load_objects, mock_load_ocr):
        image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
        ocr_keywords = ['keyword1']
        object_names = ['object1']
        object_counts = ['count1']

        mock_load_ocr.return_value = (
            {'image1.jpg': ['keyword1'], 'image2.jpg': ['keyword2'], 'image3.jpg': ['keyword1']},
            ['keyword1', 'keyword2']
        )
        mock_load_objects.return_value = (
            {'image1.jpg': ['object1'], 'image2.jpg': ['object2'], 'image3.jpg': ['object1']},
            ['object1', 'object2']
        )
        mock_load_counts.return_value = (
            {'image1.jpg': ['count1'], 'image2.jpg': ['count2'], 'image3.jpg': ['count1']},
            ['count1', 'count2']
        )

        # Execute
        filtered_paths, available_ocr_keywords, available_objects, available_object_counts = apply_metadata_filters(
            image_paths=image_paths,
            ocr_keywords=ocr_keywords,
            object_names=object_names,
            object_counts=object_counts
        )

        # Verify
        self.assertEqual(filtered_paths, ['image1.jpg', 'image3.jpg'])
        self.assertEqual(available_ocr_keywords, ['keyword1', 'keyword2'])
        self.assertEqual(available_objects, ['object1', 'object2'])
        self.assertEqual(available_object_counts, ['count1', 'count2'])

    def test_apply_metadata_filters_no_filters(self):
        # If no filters are applied, all image paths should be returned
        image_paths = ['image1.jpg', 'image2.jpg']
        ocr_keywords = []
        object_names = []
        object_counts = []

        with patch('data_loaders.metadata_loader.load_ocr_data', return_value=({}, [])), \
             patch('data_loaders.metadata_loader.load_object_data', return_value=({}, [])), \
             patch('data_loaders.metadata_loader.load_object_count_data', return_value=({}, [])):
            filtered_paths, available_ocr_keywords, available_objects, available_object_counts = apply_metadata_filters(
                image_paths=image_paths,
                ocr_keywords=ocr_keywords,
                object_names=object_names,
                object_counts=object_counts
            )
            self.assertEqual(filtered_paths, image_paths)

if __name__ == '__main__':
    unittest.main()
