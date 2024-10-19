import json
import asyncio
import tenacity
from utilities.json_parser import parse_json_response
from llm_connectors.llm_connector import LLMConnector
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    pass

class ResultValidatorAgent:
    def __init__(self, llm_connector: LLMConnector, max_concurrent_requests: int = 500):
        self.llm_connector = llm_connector
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def validate_single_result(self, image_result: Dict[str, Any], clip_prompt: str, question: Optional[str] = None) -> Dict[str, Any]:
        image_path = image_result.get('image_path', '')
        if not image_path:
            return {
                'image_path': '',
                'error': 'No image path provided',
                'match_assessment': {'category': 'No Match', 'confidence': 0},
                'justification': 'Invalid input'
            }

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
            logger.error(f"Error during validation for image {image_path}: {e}")
            return {
                'image_path': image_path,
                'error': str(e),
                'match_assessment': {
                    'category': "No Match",
                    'confidence': 0
                },
                'justification': "Validation failed due to error."
            }

    async def validate_results(self, image_results: List[Dict[str, Any]], crafted_prompts: Dict[str, Any]) -> List[Dict[str, Any]]:
        tasks = []
        for image_result in image_results:
            for prompt in crafted_prompts['clip_prompts']:
                clip_prompt = prompt['prompt']
                question = crafted_prompts.get('question')
                tasks.append(self.validate_single_result(image_result, clip_prompt, question))

        validated_results = await asyncio.gather(*tasks)
        return [result for result in validated_results if result is not None]

    def _generate_validator_prompt(self, clip_prompt, question, image_path):
        return f"""
        you're an expert image analyst. analyze this image based on the given prompt and question.

        <prompt>
        {clip_prompt}
        </prompt>
        <question>
        {question or 'No question provided'}
        </question>
        <image path>
        {image_path}
        </image path>

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
            logger.debug(f"Raw LLM response: {response}")
            validation = parse_json_response(response)
            logger.debug(f"Parsed validation: {validation}")
            # Handle 'match_assessment' being a string
            match_assessment = validation.get('match_assessment', {})
            if isinstance(match_assessment, str):
                match_assessment = {
                    'category': match_assessment,
                    'confidence': 0.0  # Default confidence
                }

            return {
                'image_path': validation.get('image_path', ''),
                'question_answer': validation.get('question_answer', {
                    'question': '',
                    'answer': '',
                    'confidence': 0.0
                }),
                'match_assessment': match_assessment,
                'justification': validation.get('justification', '')
            }
        except Exception as e:
            logger.error(f"Error parsing validation: {e}")
            # Handle parsing error
            return {
                'image_path': '',
                'question_answer': {
                    'question': '',
                    'answer': '',
                    'confidence': 0.0
                },
                'match_assessment': {
                    'category': 'No Match',
                    'confidence': 0.0
                },
                'justification': f'Failed to parse validation: {str(e)}'
            }
