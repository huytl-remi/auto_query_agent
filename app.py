# app.py
import streamlit as st

import asyncio
import os
import logging

from config import Config
from llm_connectors.llm_connector import LLMConnector
from services.agent_search_service import AgentSearchService
from agents.agent_orchestrator import AgentOrchestrator
from services.search_service import perform_search

from utilities.csv_utils import create_csv_file, create_csv_with_selected_images
from utilities.model_utils import load_model, load_faiss_index, load_id2img_fps
from utilities.utils import sanitize_filename
from utilities.ui_utils import (
    display_image_with_buttons,
    display_validation_details,
    display_surrounding_frames,
    create_image_selector
)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(layout="wide")

# Initialize session state
if 'verified_classification' not in st.session_state:
    st.session_state.verified_classification = None
if "deleted_images" not in st.session_state:
    st.session_state.deleted_images = set()
if "selected_images" not in st.session_state:
    st.session_state.selected_images = []
if 'filtered_results' not in st.session_state:
    st.session_state.filtered_results = []
if 'clicked_image_path' not in st.session_state:
    st.session_state.clicked_image_path = None
if 'agent_search_started' not in st.session_state:
    st.session_state.agent_search_started = False
if 'text_query' not in st.session_state:
    st.session_state.text_query = ''
if 'classification' not in st.session_state:
    st.session_state.classification = None
if 'agent_results' not in st.session_state:
    st.session_state.agent_results = None

def display_images(image_paths, id2img_fps):
    cols = st.columns(5)
    for i, path in enumerate(image_paths):
        with cols[i % 5]:
            display_image_with_buttons(path, id2img_fps, i)

    create_image_selector(image_paths, 'regular')

def display_validated_results(validated_results, id2img_fps):
    filtered_results = [res for res in validated_results if res['match_assessment']['category'] in ['Exact Match', 'Near Match']]
    filtered_results.sort(key=lambda x: ('1' if x['match_assessment']['category'] == 'Exact Match' else '2', -x['match_assessment']['confidence']))

    st.session_state.filtered_results = filtered_results

    if filtered_results:
        st.subheader(f"Showing {len(filtered_results)} Exact/Near Matches")
        cols = st.columns(5)
        for i, validation in enumerate(filtered_results):
            with cols[i % 5]:
                video_name, frame_idx = display_image_with_buttons(validation['image_path'], id2img_fps, i, is_validated=True)
                display_validation_details(validation, video_name, frame_idx, i)

        create_image_selector([res['image_path'] for res in filtered_results], 'validated')
    else:
        st.warning("No Exact or Near Matches found in the validated results.")

def main():
    st.title("Image Search")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        top_k = st.number_input("Enter value for top_k:", min_value=1, value=10)

    with col2:
        search_method = st.selectbox("Choose search method:", ["CLIP", "Captioning", "OCR", "Agent"])

    with col3:
        if search_method != "Agent":
            text_query = st.text_input(f"Enter your query for {search_method} search:", "")
        else:
            text_query = st.text_area("Enter the competition prompt:")

    # Initialize provider_name with a default value
    provider_name = "openai"

    with col4:
        if search_method == "Agent":
            provider_name = st.selectbox("Choose LLM provider:", ["openai",
                                                                #"anthropic",
                                                                "gemini"])

    config = Config()
    model, preprocess = load_model()
    index = load_faiss_index()
    id2img_fps = load_id2img_fps()

    if search_method != "Agent" and text_query:
        image_paths = perform_search(
            search_method=search_method,
            model=model,
            index=index,
            text_query=text_query,
            top_k=top_k,
            deleted_images=st.session_state.deleted_images,
            id2img_fps=id2img_fps
        )

        st.subheader("Search Results")
        display_images(image_paths, id2img_fps)

    elif search_method == "Agent" and text_query:
        logger.info(f"Starting Agent search with query: {text_query}")
        agent_search_service = AgentSearchService(model, index, id2img_fps)

        # Use the selected provider
        api_key = Config.get_api_key(provider_name)
        llm_connector = LLMConnector(provider_name=provider_name, api_key=api_key)
        agent_orchestrator = AgentOrchestrator(llm_connector, agent_search_service)

        if st.button("Run Agent Search") or st.session_state.agent_results is not None:
            if text_query and st.session_state.agent_results is None:
                try:
                    with st.spinner("Classifying query..."):
                        classification = asyncio.run(agent_orchestrator.query_classifier.classify_query(text_query))

                    logger.debug("Starting query processing with classification")
                    with st.spinner("Processing query..."):
                        results = asyncio.run(agent_orchestrator.process_query(text_query, top_k, classification))

                    if results:
                        logger.info(f"Query processing completed. Number of results: {len(results)}")
                        st.session_state.agent_results = results
                    else:
                        logger.warning("Query processing returned no results")
                        st.warning("No results found. Please try a different query.")
                except Exception as e:
                    logger.error(f"Error during query processing: {str(e)}", exc_info=True)
                    st.error(f"An error occurred during processing: {str(e)}")

            if st.session_state.agent_results:
                st.subheader("Search Results")
                display_validated_results(st.session_state.agent_results, id2img_fps)

        if st.button("Clear Agent Search Results"):
            st.session_state.agent_results = None
            st.rerun()

    if st.session_state.get('clicked_image_path'):
        display_surrounding_frames(st.session_state.clicked_image_path, id2img_fps)

if __name__ == "__main__":
    main()
