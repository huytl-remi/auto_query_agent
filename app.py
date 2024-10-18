# app.py
import streamlit as st
import asyncio
import os
import re  # Added for sanitization

from config import Config

from llm_connectors.llm_connector import LLMConnector

from services.search_service import perform_search
from services.filter_service import apply_metadata_filters
from services.agent_search_service import AgentSearchService

from agents.agent_orchestrator import AgentOrchestrator

from utilities.video_utils import display_video_for_frame, get_video_and_frame_idx
from utilities.csv_utils import create_csv_file, create_csv_with_selected_images
from utilities.model_utils import load_model, load_faiss_index, load_id2img_fps

from data_loaders.metadata_loader import load_ocr_data, load_object_data, load_object_count_data

from session.session_state import (
    toggle_select,
    toggle_delete,
)

from utilities.utils import sanitize_filename  # Importing sanitize_filename

llm_provider = Config.DEFAULT_LLM_PROVIDER
api_key = Config.get_api_key(llm_provider)
llm_connector = LLMConnector(provider_name=llm_provider, api_key=api_key)

st.set_page_config(layout="wide")

# Initialize session state for deleted and selected images
if "deleted_images" not in st.session_state:
    st.session_state.deleted_images = set()

if "selected_images" not in st.session_state:
    st.session_state.selected_images = []

def human_verification(classification):
    st.subheader("Human Verification")

    temporal = st.checkbox("Temporal Query", value=classification['temporal'])
    question = st.checkbox("Question-based Query", value=classification['question'])

    num_scenes = st.number_input("Number of Scenes", min_value=1, value=classification['number_of_scenes'])

    scenes = []
    for i in range(num_scenes):
        st.subheader(f"Scene {i+1}")
        scene_desc = st.text_area(f"Scene {i+1} Description", value=classification['scenes'][i]['description'] if i < len(classification['scenes']) else "")
        scene_question = st.checkbox(f"Scene {i+1} has a question", value=classification['scenes'][i]['question'] if i < len(classification['scenes']) else False)
        specific_question = st.text_input(f"Scene {i+1} Question", value=classification['scenes'][i]['specific_question'] if i < len(classification['scenes']) else "", disabled=not scene_question)

        scenes.append({
            "scene": i+1,
            "description": scene_desc,
            "question": scene_question,
            "specific_question": specific_question if scene_question else None
        })

    if st.button("Confirm Classification"):
        return {
            "temporal": temporal,
            "question": question,
            "number_of_scenes": num_scenes,
            "scenes": scenes
        }
    return None

def display_images(image_paths, id2img_fps):
    cols = st.columns(5)
    for i, path in enumerate(image_paths):
        video_name, frame_idx, time_display = get_video_and_frame_idx(path, id2img_fps)
        image_name = os.path.basename(path)  # Get the image file name like '0001.jpg'

        with cols[i % 5]:  # Display images in 5 columns per row
            st.image(path, caption=f"{i+1}. Video: {video_name}\nFrame: {frame_idx}\nTime: {time_display}")

            if st.button(f"Play Video {video_name}", key=f"play_{path}"):
                display_video_for_frame(video_name, frame_idx)

            if st.button(f"Select Image {i+1}", key=f"select_{path}"):
                toggle_select((video_name, frame_idx))

            if st.button(f"Delete Image {i+1}", key=f"delete_{path}"):
                toggle_delete(path)

def display_validated_results(validated_results, id2img_fps):
    # Filter out 'Weak Match' and 'No Match' results
    filtered_results = [res for res in validated_results if res['match_assessment']['category'] in ['Exact Match', 'Near Match']]

    # Sort results, placing 'Exact Match' at the top
    filtered_results.sort(key=lambda x: ('1' if x['match_assessment']['category'] == 'Exact Match' else '2', -x['match_assessment']['confidence']))

    # Display the results
    cols = st.columns(5)
    for i, validation in enumerate(filtered_results):
        path = validation['image_path']
        image_index = i + 1
        video_name, frame_idx, time_display = get_video_and_frame_idx(path, id2img_fps)
        confidence = validation['match_assessment']['confidence']
        category = validation['match_assessment']['category']
        justification = validation['justification']

        with cols[i % 5]:
            st.image(path, caption=f"{image_index}. Video: {video_name}\nFrame: {frame_idx}\nTime: {time_display}")
            st.write(f"Category: {category}")
            st.write(f"Confidence Score: {confidence}")
            st.write(f"Justification: {justification}")
            if validation['question_answer']['answer']:
                st.write(f"Answer: {validation['question_answer']['answer']}")
            if st.button(f"Play Video {video_name}", key=f"play_validated_{path}"):
                display_video_for_frame(video_name, frame_idx)
            if st.button(f"Select {image_index}", key=f"select_validated_{path}"):
                toggle_select((video_name, frame_idx))
            if st.button(f"Delete {image_index}", key=f"delete_validated_{path}"):
                toggle_delete(path)

def main():
    st.title("Image Search")

    # Inputs for top_k and search method
    col1, col2, col3 = st.columns(3)

    with col1:
        top_k = st.number_input("Enter value for top_k:", min_value=1, value=10)

    with col2:
        search_method = st.selectbox("Choose search method:", ["CLIP", "Captioning", "OCR", "Agent"])

    with col3:
        if search_method != "Agent":
            text_query = st.text_input(f"Enter your query for {search_method} search:", "")
        else:
            text_query = st.text_area("Enter the competition prompt:")

    # Load configurations
    config = Config()
    model, preprocess = load_model()
    index = load_faiss_index()
    id2img_fps = load_id2img_fps()

    if search_method != "Agent" and text_query:
        # Perform search
        image_paths = perform_search(
            search_method=search_method,
            model=model,
            index=index,
            text_query=text_query,
            top_k=top_k,
            deleted_images=st.session_state.deleted_images,
            id2img_fps=id2img_fps
        )

        # Apply metadata filters
        ocr_keywords = []
        object_names = []
        object_counts = []
        if search_method != "OCR":  # Assuming OCR search already incorporates OCR
            ocr_data, available_ocr_keywords = load_ocr_data(image_paths)
            object_data, available_objects = load_object_data(image_paths)
            count_data, available_object_counts = load_object_count_data(image_paths)

            col1, col2, col3 = st.columns(3)
            with col1:
                ocr_keywords = st.multiselect("Select OCR text:", available_ocr_keywords)
            with col2:
                object_names = st.multiselect("Select Object:", available_objects)
            with col3:
                object_counts = st.multiselect("Select Object Count:", available_object_counts)

            if ocr_keywords or object_names or object_counts:
                image_paths, _, _, _ = apply_metadata_filters(
                    image_paths=image_paths,
                    ocr_keywords=ocr_keywords,
                    object_names=object_names,
                    object_counts=object_counts
                )

        # Display initial search results
        st.subheader("Search Results")
        display_images(image_paths, id2img_fps)

    elif search_method == "Agent" and text_query:
        agent_search_service = AgentSearchService(model, index, id2img_fps)
        agent_orchestrator = AgentOrchestrator(llm_connector, agent_search_service)

        if st.button("Run Agent Search"):
            with st.spinner("Agent processing query..."):
                # Step 1: Query Classification
                classification = agent_orchestrator.query_classifier.classify_query(text_query)

                # Step 2: Human Verification
                st.subheader("Query Classification")
                st.json(classification)

                verified_classification = human_verification(classification)

                if verified_classification:
                    # Step 3: Process Query
                    results = asyncio.run(agent_orchestrator.process_query(text_query, verified_classification))

                    # Step 4: Display Results
                    st.subheader("Search Results")
                    display_validated_results(results, id2img_fps)

    # CSV Creation and Selected Images Display
    left_col, right_col = st.columns(2)
    with left_col:
        # CSV creation and download
        st.subheader("Create CSV File")
        x = st.number_input("Enter question number:", min_value=0, value=1)
        y = st.selectbox("Select type (kis or qa):", ['kis', 'qa'])
        video_id_input = st.text_input("Enter video_id:")
        video_id = sanitize_filename(video_id_input)  # Sanitizing input
        time_str = st.text_input("Enter time (mm:ss):")
        answer = None
        if y == 'qa':
            answer = st.number_input("Enter answer:", min_value=0, value=1)

        if st.button("Create CSV"):
            csv_content, file_path = create_csv_file(x, y, video_id, time_str, answer)
            if csv_content:
                st.download_button(label="Download CSV", data=csv_content, file_name=f"query-{x}-{y}.csv")

    with right_col:
        # Display selected images
        st.subheader("Selected Images")
        if st.session_state.selected_images:
            for img in st.session_state.selected_images:
                st.write(f"Video: {img[0]}, Frame: {img[1]}")
                if st.button(f"Remove {img[0]} - {img[1]}"):
                    st.session_state.selected_images.remove(img)

            if st.button("Create CSV with selected images"):
                csv_content = create_csv_with_selected_images(st.session_state.selected_images, y, answer)
                st.download_button(label="Download CSV", data=csv_content, file_name=f"query-{y}.csv")

if __name__ == "__main__":
    main()
