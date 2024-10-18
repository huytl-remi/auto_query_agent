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

    async def validate_single_result(self, image_result, clip_prompt, question):
        image_path = image_result.get('image_path', '')
        if not image_path:
            return None

        validator_prompt = self._generate_validator_prompt(clip_prompt, question, image_path)

        @tenacity.retry(
            wait=tenacity.wait_exponential(min=1, max=10),
            stop=tenacity.stop_after_attempt(3),
            retry=tenacity.retry_if_exception_type(Exception)
        )
        async def make_request():
            return await self.llm_connector.analyze_image(image_path, validator_prompt)

        try:
            async with self.semaphore:
                response = await make_request()
                validation = self.parse_validation(response)
                return validation
        except Exception as e:
            print(f"Error during validation for image {image_path}: {e}")
            return {
                'image_path': image_path,
                'error': str(e),
                'match_assessment': {
                    'category': "No Match",
                    'confidence': 0
                },
                'justification': "Validation failed due to error."
            }

    async def validate_results(self, image_results, crafted_prompts):
        tasks = []
        for image_result in image_results:
            for prompt in crafted_prompts['clip_prompts']:
                clip_prompt = prompt['prompt']
                question = crafted_prompts.get('question')
                tasks.append(self.validate_single_result(image_result, clip_prompt, question))

        validated_results = await asyncio.gather(*tasks)
        return validated_results

    def _generate_validator_prompt(self, clip_prompt, question, image_path):
        return f"""
        you're an expert image analyst. analyze this image based on the given prompt and question.

        prompt: {clip_prompt}
        question: {question or 'No question provided'}
        image path: {image_path}

        instructions:
        1. break down the prompt into key visual elements.
        2. for each element, determine if it's present in the image and assign a confidence score.
        3. if there's a question, answer it based on the image content.
        4. provide an overall match assessment and justification.

        output your analysis as json:
        {{
        "image_path": "path/to/image.jpg",
        "visual_elements": [
            {{
            "element": "description of visual element",
            "present": true/false,
            "confidence": 0.0 to 1.0
            }},
            ...
        ],
        "question_answer": {{
            "question": "the question if provided, otherwise null",
            "answer": "your answer or null if no question",
            "confidence": 0.0 to 1.0
        }},
        "match_assessment": {{
            "category": "Exact Match" / "Near Match" / "Weak Match" / "No Match",
            "confidence": 0.0 to 1.0
        }},
        "justification": "brief explanation of your assessment"
        }}

        important:
        - be thorough and confident in your analysis.
        - confidence scores should reflect your certainty.
        """

    def parse_validation(self, response):
        try:
            validation = parse_json_response(response)

            required_keys = ['image_path', 'question_answer', 'match_assessment', 'justification']
            if not all(key in validation for key in required_keys):
                raise KeyError("Missing required keys in the validation response.")

            # Validate field types and structures
            if not isinstance(validation['image_path'], str):
                raise TypeError(f"'image_path' should be a string. Got {type(validation['image_path'])}.")

            if not isinstance(validation['question_answer'], dict):
                raise TypeError(f"'question_answer' should be a dictionary. Got {type(validation['question_answer'])}.")

            question_answer = validation['question_answer']
            if not all(key in question_answer for key in ['question', 'answer', 'confidence']):
                raise KeyError("Missing required keys in question_answer.")

            if not isinstance(question_answer['confidence'], (int, float)) or not 0 <= question_answer['confidence'] <= 1:
                raise ValueError(f"'confidence' in question_answer should be a number between 0 and 1. Got {question_answer['confidence']}.")

            if not isinstance(validation['match_assessment'], dict):
                raise TypeError(f"'match_assessment' should be a dictionary. Got {type(validation['match_assessment'])}.")

            if not all(key in validation['match_assessment'] for key in ['category', 'confidence']):
                raise KeyError("Missing required keys in match_assessment.")

            if validation['match_assessment']['category'] not in ["Exact Match", "Near Match", "Weak Match", "No Match"]:
                raise ValueError(f"Invalid category: {validation['match_assessment']['category']}")

            if not isinstance(validation['match_assessment']['confidence'], (int, float)) or not 0 <= validation['match_assessment']['confidence'] <= 1:
                raise ValueError(f"'confidence' in match_assessment should be a number between 0 and 1. Got {validation['match_assessment']['confidence']}.")

            if not isinstance(validation['justification'], str):
                raise TypeError(f"'justification' should be a string. Got {type(validation['justification'])}.")

            # Return only the required fields
            return {
                'image_path': validation['image_path'],
                'question_answer': {
                    'question': question_answer['question'],
                    'answer': question_answer['answer'],
                    'confidence': question_answer['confidence']
                },
                'match_assessment': {
                    'category': validation['match_assessment']['category'],
                    'confidence': validation['match_assessment']['confidence']
                },
                'justification': validation['justification']
            }

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            print(f"Error parsing validation: {e}")
            raise ValueError("Failed to parse or validate the LLM response correctly.")
