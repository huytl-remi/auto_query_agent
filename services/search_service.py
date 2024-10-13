# services/search_service.py
from utilities.model_utils import (
    search_image_by_text,
    search_image_by_text_with_captioning,
    search_images_by_ocr,
    get_image_paths,
)

def perform_search(search_method, model, index, text_query, top_k, deleted_images, id2img_fps):
    """
    Perform image search based on the selected method.

    Args:
        search_method (str): The search method ("CLIP", "Captioning", "OCR").
        model: The loaded CLIP model.
        index: The FAISS index.
        text_query (str): The user's search query.
        top_k (int): Number of top results to retrieve.
        deleted_images (set): Set of image paths marked as deleted.
        id2img_fps (dict): Mapping from IDs to image file paths.

    Returns:
        list: List of image paths matching the search criteria.
    """
    if search_method == "CLIP":
        image_indices, distances = search_image_by_text(model, index, text_query, top_k + len(deleted_images))
        image_paths = get_image_paths(image_indices, id2img_fps)
    elif search_method == "Captioning":
        image_paths = search_image_by_text_with_captioning(text_query, top_k)
    elif search_method == "OCR":
        image_paths = search_images_by_ocr(text_query, top_k)
    else:
        raise ValueError(f"Unsupported search method: {search_method}")

    # Filter out deleted images
    image_paths = [path for path in image_paths if path not in deleted_images]

    return image_paths
