# agents/result_validator_agent.pyimport json
import json
import asyncio
import tenacity
from utilities.json_parser import parse_json_response
from llm_connectors.llm_connector import LLMConnector

from config import Config

llm_provider = Config.DEFAULT_LLM_PROVIDER
api_key = Config.get_api_key(llm_provider)
llm_connector = LLMConnector(provider_name=llm_provider, api_key=api_key)

class ResultValidatorAgent:
    def __init__(self, llm_connector, max_concurrent_requests=100):
        self.llm_connector = llm_connector
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def validate_single_result(self, image_result, query):
        image_path = image_result.get('image_path', '')
        if not image_path:
            return None

        prompt = self.generate_prompt(image_path, query)

        @tenacity.retry(
            wait=tenacity.wait_exponential(min=1, max=10),
            stop=tenacity.stop_after_attempt(3),
            retry=tenacity.retry_if_exception_type(Exception)
        )
        async def make_request():
            return await self.llm_connector.analyze_image(image_path, prompt)

        try:
            async with self.semaphore:
                response = await make_request()
                validation = self.parse_validation(response)
                return validation
        except Exception as e:
            print(f"Error during validation for image {image_path}: {e}")
            return {
                'image': image_path,
                'error': str(e),
                'category': "No Match",
                'confidence_score': 0,
                'justification': "Validation failed due to error."
            }

    async def validate_results(self, image_results, query):
        tasks = [self.validate_single_result(image_result, query) for image_result in image_results]
        validated_results = await asyncio.gather(*tasks)
        return validated_results

    def generate_prompt(self, image_path, query):
        prompt = f"""
        You are an expert image analyst.

        **Task**:
        Analyze the image and determine how well it matches the following query.

        **Query**: {query}
        **Image Path**: {image_path}

        **Instructions**:
        1. **Compare** the image to the query.
        2. **Provide** a **confidence score** between 0 and 100 indicating how well the image matches the query.
        3. **Categorize** the image into one of the following categories:
        - **Exact Match**: The image perfectly matches all aspects of the query.
        - **Near Match**: The image closely matches the query but may have minor differences.
        - **Weak Match**: The image has some elements matching the query but significant details are missing.
        - **No Match**: The image does not match the query.
        4. **Justify** your categorization with a brief explanation (one or two sentences).

        **Output Format**:
        Provide a JSON object with the following structure:

        {{
        "image": "<full_image_path>",
        "category": "<Exact Match/Near Match/Weak Match/No Match>",
        "confidence_score": <number between 0 and 100>,
        "justification": "<brief explanation>"
        }}

        **Important**:
        - Only include the requested JSON object in your response.
        - Ensure all numeric values are numbers (not strings).
        - Do not add any additional commentary or explanations.
        """
        return prompt

    def parse_validation(self, response):
        # Parse the LLM's response and validate its structure
        try:
            validation = parse_json_response(response)

            if not all(key in validation for key in ['image', 'category', 'confidence_score', 'justification']):
                raise KeyError("Missing required keys in the validation response.")

            # Validate field types
            if not isinstance(validation['image'], str):
                raise TypeError(f"'image' should be a string. Got {type(validation['image'])}.")
            if not isinstance(validation['category'], str):
                raise TypeError(f"'category' should be a string. Got {type(validation['category'])}.")
            if not isinstance(validation['confidence_score'], (int, float)):
                raise TypeError(f"'confidence_score' should be a number. Got {type(validation['confidence_score'])}.")
            if not isinstance(validation['justification'], str):
                raise TypeError(f"'justification' should be a string. Got {type(validation['justification'])}.")

            return validation
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error parsing validation: {e}")
            raise ValueError("Failed to parse or validate the LLM response correctly.")
