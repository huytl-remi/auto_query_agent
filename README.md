# Image Query Validator

## Overview

This application allows users to query a vector database containing images, either encoded by CLIP or embedded from captions using vision language models. It provides a streamlined interface for querying, validating, and exporting images based on user queries.

## Features

- **Query Methods**:
  - CLIP search
  - Captioning search
  - OCR search
- **AI Validation**:
  - Option to validate top results using an AI agent powered by various LLMs (OpenAI, Anthropic, Gemini).
  - The validation process categorizes results as `Exact Match`, `Near Match`, `Weak Match`, or `No Match` with confidence scores.
- **Video and Frame Management**:
  - Display videos corresponding to image results.
  - Temporal frame navigation for surrounding images.
- **CSV Export**:
  - Generate CSV files based on selected images and query results.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/image-query-validator.git
   cd image-query-validator
   ```

2. Set up a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the root directory of the project.
   - Add the following variables to the `.env` file:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ANTHROPIC_API_KEY=your_anthropic_api_key_here
     GEMINI_API_KEY=your_gemini_api_key_here
     PINECONE_API_KEY=your_pinecone_api_key_here

     DEFAULT_LLM_PROVIDER=openai
     OPENAI_MODEL=gpt-4
     ANTHROPIC_MODEL=claude-2
     GEMINI_MODEL=gemini-1.5-flash
     ```
   - Replace the placeholder values with your actual API keys.

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## File Structure

- **app.py**: Main application file for querying and interacting with the interface.
- **agents/**: Contains the AI validation agent.
- **llm_connectors/**: Interfaces for various LLM providers.
- **data_loaders/**: Functions for loading image metadata (OCR, objects, counts).
- **utilities/**: Helper functions for model loading, video handling, and CSV generation.
- **session/**: Manages session state for selected and deleted images.
- **config.py**: Centralized configuration management using environment variables.
- **requirements.txt**: List of project dependencies.

## Query Process

1. Enter a natural language query and select the search method (CLIP, Captioning, OCR).
2. Review the top `P` results.
3. (Optional) Click "Run AI Validator" to refine results based on AI-powered analysis.
4. Filter and export the results as needed.

## Dependencies

The main dependencies for this project are listed in the `requirements.txt` file. Key libraries include:

- streamlit
- torch
- CLIP
- FAISS
- OpenAI
- Pinecone
- Google GenerativeAI
- Anthropic
- pandas

Refer to `requirements.txt` for the full list of dependencies and their versions.

## Configuration

The project uses a `.env` file for managing API keys and default settings. The `config.py` file loads these environment variables and provides a centralized way to access them throughout the application.

To change the default LLM provider or model, update the corresponding variables in the `.env` file.

## Contributing

[Add your contribution guidelines here]

## License

[Add your license information here]
