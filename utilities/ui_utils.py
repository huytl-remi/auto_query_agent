# utilities/ui_utils.py

import streamlit as st
import os
from utilities.video_utils import get_temporal_frames, get_video_and_frame_idx, display_video_for_frame
from utilities.model_utils import load_id2img_fps
from session.session_state import toggle_select, toggle_delete

id2img_fps = load_id2img_fps()

def display_image_with_buttons(path, id2img_fps, index, is_validated=False):
    video_name, frame_idx, time_display, mili = get_video_and_frame_idx(path, id2img_fps)

    st.image(path, caption=f"{index+1}. {video_name}, Time: {time_display}, Mili: {mili}s")

    if not is_validated:
        if st.button(f"Play Video {video_name}", key=f"play_{path}"):
            display_video_for_frame(video_name, frame_idx)

        if st.button(f"Select Image {index+1}", key=f"select_{path}"):
            toggle_select((video_name, frame_idx))

        if st.button(f"Delete Image {index+1}", key=f"delete_{path}"):
            toggle_delete(path)

    return video_name, frame_idx

def display_validation_details(validation, video_name, frame_idx, index):
    confidence = validation['match_assessment']['confidence']
    category = validation['match_assessment']['category']
    justification = validation['justification']

    st.write(f"Category: {category}")
    st.write(f"Confidence: {confidence:.2f}")
    st.write(f"Justification: {justification}")
    if validation.get('question_answer', {}).get('answer'):
        st.write(f"Answer: {validation['question_answer']['answer']}")
    if st.button(f"Play Video {video_name}", key=f"play_validated_{validation['image_path']}"):
        display_video_for_frame(video_name, frame_idx)
    if st.button(f"Select {index+1}", key=f"select_validated_{validation['image_path']}"):
        toggle_select((video_name, frame_idx))
    if st.button(f"Delete {index+1}", key=f"delete_validated_{validation['image_path']}"):
        toggle_delete(validation['image_path'])

def display_surrounding_frames(clicked_image_path, id2img_fps):
    if clicked_image_path:
        st.image(clicked_image_path, caption=f"Selected Image: {os.path.basename(clicked_image_path)}", use_column_width=True)
        surrounding_frames = get_temporal_frames(clicked_image_path, min_distance=50, temporal_range=5)
        st.subheader("Surrounding Frames")
        surrounding_cols = st.columns(5)
        for i, frame_path in enumerate(surrounding_frames):
            surr_video_name, surr_frame_idx, surr_time_display, surr_mili = get_video_and_frame_idx(frame_path, id2img_fps)
            with surrounding_cols[i % 5]:
                st.image(frame_path, caption=f"{i+1}. {surr_video_name}, {os.path.basename(frame_path)}, Mili:{surr_mili}s ", use_column_width=True)
                if st.button(f"Select Surr {i+1}", key=f"select_surr_{frame_path}"):
                    toggle_select((surr_video_name, surr_frame_idx))
                if st.button(f"Delete Surr {i+1}", key=f"delete_surr_{frame_path}"):
                    toggle_delete(frame_path)

def create_image_selector(image_paths, key_prefix):
    st.subheader("Click on an image to see surrounding frames")
    image_paths_with_stt = [f"{i+1}. {path}" for i, path in enumerate(image_paths)]
    clicked_image_path_with_stt = st.selectbox("Select an image to see surrounding frames:",
                                               image_paths_with_stt,
                                               key=f'{key_prefix}_image_selector')
    if clicked_image_path_with_stt:
        clicked_image_path = clicked_image_path_with_stt.split(". ", 1)[1]
        st.session_state.clicked_image_path = clicked_image_path

    if st.session_state.get('clicked_image_path'):
        if st.button("Display Surrounding Frames", key=f'{key_prefix}_display_surrounding_frames_button'):
            display_surrounding_frames(st.session_state.clicked_image_path, id2img_fps)
