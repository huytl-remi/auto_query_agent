# utilities/video_utils.py
import os
import subprocess
import pandas as pd
import streamlit as st
import re  # Added for sanitization

from utilities.utils import sanitize_filename  # Importing sanitize_filename

def generate_unique_file_name(base_name, extension):
    """Generate a unique file name to avoid overwriting existing files."""
    counter = 1
    new_file_name = f"{base_name}{extension}"

    # Loop to find an available file name by appending a counter to the base name
    while os.path.exists(new_file_name):
        new_file_name = f"{base_name}_{counter}{extension}"
        counter += 1

    return new_file_name

# Extract video segment based on time
def find_video_file(base_path, video_part, video_id):
    """
    Search for the video file in the two possible locations.
    """
    path_with_video_folder = os.path.join(base_path, f"Videos_{video_part}", "video", f"{video_id}.mp4")
    path_without_video_folder = os.path.join(base_path, f"Videos_{video_part}", f"{video_id}.mp4")

    if os.path.exists(path_with_video_folder):
        return path_with_video_folder
    elif os.path.exists(path_without_video_folder):
        return path_without_video_folder
    else:
        return None

def extract_video_segment(video_id, start_time, end_time):
    video_id = sanitize_filename(video_id)
    video_part = video_id.split('_')[0]
    base_path = "/content/drive/MyDrive/HCMC_AI/AIC_Video"

    video_path = find_video_file(base_path, video_part, video_id)

    if not video_path:
        error_message = f"Video file not found for ID: {video_id}"
        return None, error_message

    base_output_file = f"{video_id}_segment"
    output_video_file = generate_unique_file_name(base_output_file, ".mp4")
    output_video_file = sanitize_filename(output_video_file)

    command = [
        "ffmpeg", "-ss", str(start_time), "-to", str(end_time),
        "-i", video_path, "-c", "copy", output_video_file
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return output_video_file, None
    except subprocess.CalledProcessError as e:
        error_message = f"Error executing ffmpeg: {e.stderr}"
        return None, error_message

# Function to display the video for a specific frame
def display_video_for_frame(video_name, frame_time):
    # Assume frame_time is in seconds (calculated from frame_idx)
    start_time = max(0, frame_time - 60)  # Start 60 seconds before the frame
    end_time = frame_time + 60  # End 60 seconds after the frame

    # Call the function to extract the video segment for this time range
    output_video_file, error = extract_video_segment(video_name, start_time, end_time)

    if error:
        st.error(f"Error extracting video segment: {error}")
    elif output_video_file:
        # Display the video on the Streamlit app
        st.video(output_video_file)

        # Now calculate the actual start and end times in minutes and seconds
        actual_start_minutes = int(start_time // 60)  # Minutes part
        actual_start_seconds = int(start_time % 60)   # Seconds part
        actual_start_time = actual_start_minutes * 60 + actual_start_seconds
        actual_end_minutes = int(end_time // 60)      # Minutes part
        actual_end_seconds = int(end_time % 60)       # Seconds part
        actual_end_time = actual_end_minutes * 60 + actual_end_seconds

        # Display the corrected time range
        st.write(f"Video is playing from {actual_start_time}s "
                 f"to {actual_end_time}s.")
    else:
        st.error("Unable to extract and display the video segment.")

# Get video ID and frame index from the image path
def get_video_and_frame_idx(image_path, id2img_fps):
    parts = image_path.split('/')
    data_part = parts[-3]
    video_id = parts[-2]
    frame_number_str = parts[-1].split('.')[0]
    frame_number = int(frame_number_str)

    csv_path = f"/content/drive/MyDrive/HCMC_AI/data/map-keyframes/{data_part.split('_')[0]}_{video_id}.csv"

    df = pd.read_csv(csv_path)
    fps = df['fps'].iloc[0]

    if 'extra' in data_part:
        frame_idx = frame_number
    else:
        frame_idx = df.loc[df['n'] == frame_number, 'frame_idx'].values[0]

    time_in_seconds = frame_idx / fps
    mili_seconds = int(time_in_seconds * 1000)
    minutes = int(time_in_seconds // 60)
    seconds = int(time_in_seconds % 60)
    time_display = f"{minutes}p{seconds}s ({int(time_in_seconds)}s)"

    return f"{data_part.split('_')[0]}_{video_id}", time_in_seconds, time_display, mili_seconds

# Get surrounding frames for a specific image
def get_temporal_frames(image_path, min_distance=25, temporal_range=5):
    dir_path = os.path.dirname(image_path)
    all_images = sorted([img for img in os.listdir(dir_path) if img.endswith('.jpg')])
    selected_image_name = os.path.basename(image_path)
    selected_index = all_images.index(selected_image_name)

    surrounding_frames = []
    count_before, count_after = 0, 0
    last_frame_number = int(selected_image_name.split('.')[0])

    for i in range(selected_index - 1, -1, -1):
        current_frame_number = int(all_images[i].split('.')[0])
        if last_frame_number - current_frame_number >= min_distance:
            surrounding_frames.insert(0, os.path.join(dir_path, all_images[i]))
            last_frame_number = current_frame_number
            count_before += 1
        if count_before == temporal_range:
            break

    last_frame_number = int(selected_image_name.split('.')[0])
    for i in range(selected_index + 1, len(all_images)):
        current_frame_number = int(all_images[i].split('.')[0])
        if current_frame_number - last_frame_number >= min_distance:
            surrounding_frames.append(os.path.join(dir_path, all_images[i]))
            last_frame_number = current_frame_number
            count_after += 1
        if count_after == temporal_range:
            break

    return surrounding_frames
