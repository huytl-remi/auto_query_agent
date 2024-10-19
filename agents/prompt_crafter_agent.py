import json
import re
from utilities.json_parser import parse_json_response
import logging

logger = logging.getLogger(__name__)

class PromptCrafterAgent:
    def __init__(self, llm_connector):
        self.llm_connector = llm_connector

    async def craft_prompts(self, classification):
        logger.info(f"Starting prompt crafting with classification: {classification}")
        prompt = self._generate_super_prompt(classification)
        logger.debug(f"Generated super prompt: {prompt}")

        response = await self.llm_connector.generate_text(prompt)
        logger.info(f"Raw LLM response:\n{response}")

        parsed_response = self._parse_response(response)
        if parsed_response is None:
            logger.error("Failed to parse LLM response")
            raise ValueError("Failed to parse LLM response")

        logger.info(f"Parsed response: {parsed_response}")
        return parsed_response

    def _generate_super_prompt(self, classification):
        return f"""
        you are an expert ai assistant specializing in crafting search prompts for a sophisticated multi-modal image retrieval system. your task is to create optimal prompts based on classified user queries.

        system capabilities:
        1. clip-based search: identifies images using concise english descriptions of visual elements.
        2. caption-based search: finds images using Vietnamese descriptions.
        3. temporal reasoning: understands sequences of events across multiple frames.
        4. question-answering: extracts specific textual or numerical information from images.

        input:
        you will receive a json object containing the query classification. example:
        {classification}

        task:
        craft search prompts optimized for both clip and caption-based searches, considering the query's temporal and question-based nature.

        guidelines:
        1. CLIP prompts (english):
           - use concise, descriptive language focusing on visual elements.
           - prioritize nouns, adjectives, and action verbs.
           - example: "man in red shirt running through crowded street"
           - Exclude text elements from the prompt

        2. caption prompts (vietnamese):
           - exact translation of CLIP prompt
           - include text element in the prompt
           - example: "Người đàn ông mặc áo đỏ có chữ 'Bon Jovi' chạy qua đường phố đông đúc"

        3. for both prompt types:
           - be specific about colors, numbers, and spatial relationships.
           - avoid subjective interpretations or non-visual information.

        4. temporal queries:
           - create separate prompts for each scene.
           - include all relevant details from previous scenes.
           - example:
             scene 1: "a woman in yellow dress entering bank"
             scene 2: "a woman in yellow dress at bank counter, holding document"

        5. question-based queries:
           - focus prompts on the visual scene related to the question.
           - do not include the question itself in the prompts.
           - example for "What's the amount of fine on the paper?":
             prompt: "A man in orange suit, with tattoo arms, is writing on paper."

        6. edge cases:
           - for very short queries, expand on likely visual context.
           - for complex queries, break down into key visual components.

        output format:
        provide a json object with the following structure:
        {{
          "clip_prompts": [
            {{
              "scene": <integer>,
              "prompt": "<concise english description>"
            }},
            ...
          ],
          "caption_prompts": [
            {{
              "scene": <integer>,
              "prompt": "<detailed vietnamese description>"
            }},
            ...
          ],
          "question": "<specific question if present, otherwise null>"
        }}

        important:
        - ensure all prompts are relevant to the query classification.
        - provide only the json object, no additional explanations.
        - use proper json formatting and escaping.
        """

    def _parse_response(self, response):
        try:
            parsed_response = parse_json_response(response)
            logger.debug(f"JSON parsed response: {parsed_response}")

            if "error" in parsed_response:
                logger.error(f"Error in LLM response: {parsed_response['error']}")
                raise ValueError(f"Error in LLM response: {parsed_response['error']}")

            required_keys = ['clip_prompts', 'caption_prompts', 'question']
            if not all(key in parsed_response for key in required_keys):
                missing_keys = [key for key in required_keys if key not in parsed_response]
                logger.error(f"Missing required keys in the response: {missing_keys}")
                raise ValueError(f"Missing required keys in the response: {missing_keys}")

            return parsed_response

        except ValueError as e:
            logger.error(f"ValueError during JSON parsing: {str(e)}")
            logger.debug(f"Raw response that caused ValueError: {response}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in parsing response: {str(e)}")
            logger.debug(f"Raw response that caused unexpected error: {response}")
            return None
