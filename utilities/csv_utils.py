import csv
import io
import pandas as pd
import os
import streamlit as st

# Calculate frame index from time and fps
def calculate_frame_idx(time_str, fps):
    minutes, seconds = map(int, time_str.split(':'))
    total_seconds = minutes * 60 + seconds
    return total_seconds * fps

# Create CSV content
def create_csv_content(x, y, video_id, base_time, answer=None):
    csv_path = f"/content/drive/MyDrive/HCMC_AI/data/map-keyframes/{video_id}.csv"
    try:
        df = pd.read_csv(csv_path)
        fps = df['fps'].iloc[0]
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None, None

    base_frame_idx = calculate_frame_idx(base_time, fps)
    frame_indices = [base_frame_idx + ((i // 2) * (-1 if i % 2 else 1) * 15) for i in range(100)]

    output = io.StringIO()
    writer = csv.writer(output)

    if y == 'kis':
        for frame_idx in frame_indices:
            writer.writerow([video_id, frame_idx])
    elif y == 'qa':
        for frame_idx in frame_indices:
            writer.writerow([video_id, frame_idx, answer])

    csv_content = output.getvalue()
    output.close()

    return csv_content

# Create CSV file
def create_csv_file(x, y, video_id, base_time, answer=None):
    csv_path = f"/content/drive/MyDrive/HCMC_AI/data/map-keyframes/{video_id}.csv"
    try:
        df = pd.read_csv(csv_path)
        fps = df['fps'].iloc[0]
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None, None

    base_frame_idx = calculate_frame_idx(base_time, fps)
    frame_indices = [base_frame_idx + ((i // 2) * (-1 if i % 2 else 1) * 15) for i in range(100)]

    file_path = f"Submission/query-{x}-{y}.csv"
    output = io.StringIO()
    writer = csv.writer(output)

    with open(file_path, mode='w', newline='') as file:
        file_writer = csv.writer(file)
        if y == 'kis':
            for frame_idx in frame_indices:
                writer.writerow([video_id, frame_idx])
                file_writer.writerow([video_id, frame_idx])
        elif y == 'qa':
            for frame_idx in frame_indices:
                writer.writerow([video_id, frame_idx, answer])
                file_writer.writerow([video_id, frame_idx, answer])

    csv_content = output.getvalue()
    output.close()

    return csv_content, file_path

# Create CSV with selected images
def create_csv_with_selected_images(selected_images, y, answer=None):
    frame_indices = []
    for video_id, frame_idx in selected_images:
        frame_indices.append((video_id, frame_idx))

    while len(frame_indices) < 100:
        for video_id, frame_idx in selected_images:
            frame_indices.append((video_id, frame_idx + 15))
            frame_indices.append((video_id, frame_idx - 15))
            if len(frame_indices) >= 100:
                break

    output = io.StringIO()
    writer = csv.writer(output)
    for i, (video_id, frame_idx) in enumerate(frame_indices[:100]):
        if y == 'kis':
            writer.writerow([video_id, frame_idx])
        elif y == 'qa':
            writer.writerow([video_id, frame_idx, answer])

    return output.getvalue()
