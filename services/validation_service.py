# services/validation_service.py
from agents.result_validator_agent import ResultValidatorAgent
import streamlit as st
import asyncio
# import time

async def run_ai_validator(image_paths, query, default_llm_provider, config):
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
    image_results = [{'image_path': path} for path in image_paths]

    from llm_connectors.llm_connector import LLMConnector
    llm_connector = LLMConnector(
        provider_name=default_llm_provider,
        api_key=config.get_api_key(default_llm_provider)
    )

    validator_agent = ResultValidatorAgent(llm_connector)

    # Start measuring time
    # start_time = time.time()

    try:
        validated_results = await validator_agent.validate_results(image_results, query)
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        validated_results = []

    # End measuring time
    # end_time = time.time()
    # elapsed_time = end_time - start_time

    # Log or display the time taken
    # st.write(f"AI validation completed in {elapsed_time:.2f} seconds")

    return validated_results
