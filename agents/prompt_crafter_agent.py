import json
from utilities.json_parser import parse_json_response

class PromptCrafterAgent:
    def __init__(self, llm_connector):
        self.llm_connector = llm_connector

    def craft_prompts(self, classification):
        prompt = self._generate_super_prompt(classification)
        response = self.llm_connector.generate_text(prompt)
        return self._parse_response(response)

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

        2. caption prompts (vietnamese):
           - exact translation of CLIP prompt
           - example: "Người đàn ông mặc áo đỏ chạy qua đường phố đông đúc"

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
            parsed_response = json.loads(response)
            if "error" in parsed_response:
                raise ValueError(f"Error in LLM response: {parsed_response['error']}")
            if not all(key in parsed_response for key in ['clip_prompts', 'caption_prompts', 'question']):
                raise ValueError("Missing required keys in the response")
            return parsed_response
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON response: {e}")
            return None
        except ValueError as e:
            print(f"Error: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return None
