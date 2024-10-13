# services/filter_service.py
from data_loaders.metadata_loader import (
    load_ocr_data,
    load_object_data,
    load_object_count_data,
)

def apply_metadata_filters(image_paths, ocr_keywords, object_names, object_counts):
    """
    Apply OCR and metadata filters to the list of image paths.

    Args:
        image_paths (list): List of image file paths.
        ocr_keywords (list): List of OCR keywords to filter by.
        object_names (list): List of object names to filter by.
        object_counts (list): List of object counts to filter by.

    Returns:
        list: Filtered list of image paths.
    """
    # Load metadata
    ocr_data, available_ocr_keywords = load_ocr_data(image_paths)
    object_data, available_objects = load_object_data(image_paths)
    count_data, available_object_counts = load_object_count_data(image_paths)

    # Apply OCR filters
    if ocr_keywords:
        image_paths = [
            path for path in image_paths
            if any(keyword.lower() in ' '.join(ocr_data.get(path, [])).lower() for keyword in ocr_keywords)
        ]

    # Apply object name and count filters
    if object_names or object_counts:
        image_paths = [
            path for path in image_paths
            if (all(obj in object_data.get(path, []) for obj in object_names) and
                all(cnt in count_data.get(path, []) for cnt in object_counts))
        ]

    return image_paths, available_ocr_keywords, available_objects, available_object_counts
