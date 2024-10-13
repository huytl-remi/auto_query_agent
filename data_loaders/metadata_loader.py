import os
import json
import streamlit as st

@st.cache_data
def load_ocr_data(image_paths):
    ocr_keywords = set()
    ocr_data = {}
    for path in image_paths:
        parts = path.split('/')
        data_part = parts[-3]
        video_id = parts[-2]
        frame_number_str = parts[-1].split('.')[0]

        ocr_file = f"/content/drive/MyDrive/HCMC_AI/data/ocr/{data_part}/{video_id}.json"
        if not os.path.exists(ocr_file):
            continue

        with open(ocr_file, "r") as file:
            ocr_json = json.load(file)
            ocr_data[path] = ocr_json.get(frame_number_str, [])
            ocr_keywords.update(ocr_json.get(frame_number_str, []))

    return ocr_data, sorted(ocr_keywords)

@st.cache_data
def load_object_data(image_paths):
    object_classes = set()
    object_data = {}
    for path in image_paths:
        parts = path.split('/')
        data_part = parts[-3]
        video_id = parts[-2]
        frame_number_str = parts[-1].split('.')[0]

        object_file = f"/content/drive/MyDrive/HCMC_AI/data/context_encoded/classes_encoded/{data_part}/{video_id}.txt"
        if not os.path.exists(object_file):
            continue

        with open(object_file, "r") as file:
            for line in file:
                if line.startswith(frame_number_str):
                    objects = line.strip().split()[1:]
                    object_data[path] = objects
                    object_classes.update(objects)
                    break
    return object_data, sorted(object_classes)

@st.cache_data
def load_object_count_data(image_paths):
    object_counts = set()
    count_data = {}
    for path in image_paths:
        parts = path.split('/')
        data_part = parts[-3]
        video_id = parts[-2]
        frame_number_str = parts[-1].split('.')[0]

        count_file = f"/content/drive/MyDrive/HCMC_AI/data/context_encoded/number_encoded/{data_part}/{video_id}.txt"
        if not os.path.exists(count_file):
            continue

        with open(count_file, "r") as file:
            for line in file:
                if line.startswith(frame_number_str):
                    counts = line.strip().split()[1:]
                    count_data[path] = counts
                    object_counts.update(counts)
                    break
    return count_data, sorted(object_counts)

# Filter images by OCR
def filter_images_by_ocr(image_paths, ocr_data, selected_ocr_keywords):
    filtered_paths = []
    for path in image_paths:
        ocr_texts = ocr_data.get(path, [])
        if any(keyword.lower() in ' '.join(ocr_texts).lower() for keyword in selected_ocr_keywords):
            filtered_paths.append(path)
    return filtered_paths

# Filter images by object metadata
def filter_images_by_metadata(image_paths, object_metadata, number_metadata, selected_objects, selected_numbers):
    filtered_paths = []
    for path in image_paths:
        objects = object_metadata.get(path, [])
        numbers = number_metadata.get(path, [])

        if all(obj in objects for obj in selected_objects) and all(num in numbers for num in selected_numbers):
            filtered_paths.append(path)
    return filtered_paths
