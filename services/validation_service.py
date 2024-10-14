# services/validation_service.py
from agents.result_validator_agent import ResultValidatorAgent
import streamlit as st

def run_ai_validator(image_paths, query, default_llm_provider, config):
    """
    Run AI validation on the list of image paths.

    Args:
        image_paths (list): List of image file paths.
        query (str): The original query string to validate the images against.
        default_llm_provider (str): The default LLM provider name.
        config (Config): The configuration object.

    Returns:
        list: List of validated results with categories and confidence scores.
    """
    # Prepare the image results
    image_results = [{'image_path': path} for path in image_paths]

    # Initialize the LLMConnector
    from llm_connectors.llm_connector import LLMConnector
    llm_connector = LLMConnector(
        provider_name=default_llm_provider,
        api_key=config.get_api_key(default_llm_provider)
    )

    # Initialize the ResultValidatorAgent
    validator_agent = ResultValidatorAgent(llm_connector)

    # Validate the results and pass the query to the agent
    try:
        validated_results = validator_agent.validate_results(image_results, query)
    except Exception as e:
        st.error(f"Error during validation: {str(e)}")
        validated_results = []

    return validated_results
