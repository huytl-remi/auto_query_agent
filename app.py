# app.py
import streamlit as st
from llm_connectors.llm_connector import LLMConnector
from main import Orchestrator

def main():
    st.title("Image Retrieval System")

    # Configuration inputs
    provider_name = st.selectbox("Select LLM Provider", ["OpenAI", "Anthropic", "Gemini"])
    api_key = st.text_input("Enter API Key", type="password")
    model_name = st.text_input("Enter Model Name (optional)")

    if st.button("Initialize LLM Connector"):
        if not api_key:
            st.error("API Key is required.")
            return

        try:
            # Instantiate the LLMConnector
            llm_connector = LLMConnector(provider_name, api_key, model_name)
            st.success(f"LLM Connector initialized with provider: {provider_name}")

            # Instantiate the Orchestrator
            orchestrator = Orchestrator(llm_connector)

            # Input query from user
            input_query = st.text_input("Enter your query:")

            if st.button("Run Query"):
                if input_query:
                    results = orchestrator.run_query(input_query)
                    if results:
                        st.write("Final Results:")
                        display_results(results)
                    else:
                        st.write("No satisfactory results found.")
                else:
                    st.write("Please enter a query.")
        except Exception as e:
            st.error(f"Failed to initialize LLM Connector: {e}")

def display_results(validated_results):
    for scene_validation in validated_results:
        scene_number = scene_validation.get('scene', 1)
        st.write(f"#### Scene {scene_number} Results")

        if 'error' in scene_validation:
            st.write(f"Error in validation: {scene_validation['error']}")
            continue

        # Display results with confidence scores and justifications
        results = scene_validation.get('results', [])
        if results:
            for result in results:
                image = result.get('image')
                category = result.get('category')
                confidence_score = result.get('confidence_score')
                justification = result.get('justification')
                st.write(f"- **Image**: {image}")
                st.write(f"  - **Category**: {category}")
                st.write(f"  - **Confidence Score**: {confidence_score}")
                st.write(f"  - **Justification**: {justification}")
                if 'answer' in result and result['answer']:
                    st.write(f"  - **Answer**: {result['answer']}")

            # Display question answer if available
            question_answer = scene_validation.get('question_answer')
            if question_answer:
                st.write(f"**Answer to the question**: {question_answer}")
        else:
            st.write("No results after validation.")

if __name__ == "__main__":
    main()
