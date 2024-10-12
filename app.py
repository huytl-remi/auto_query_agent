import streamlit as st

from llm_connector import LLMConnector
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
        except Exception as e:
            st.error(f"Failed to initialize LLM Connector: {e}")
            return

        # Instantiate the Orchestrator
        orchestrator = Orchestrator(llm_connector)

        # Input query from user
        input_query = st.text_input("Enter your query:")

        if st.button("Run Query"):
            if input_query:
                results = orchestrator.run_query(input_query)
                if results:
                    st.write("Final Results:")
                    st.write(results)
                else:
                    st.write("No satisfactory results found.")
            else:
                st.write("Please enter a query.")

if __name__ == "__main__":
    main()
