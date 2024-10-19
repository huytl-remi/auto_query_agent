# agents/query_classifier_agent.py
import json
from utilities.json_parser import parse_json_response

class QueryClassifierAgent:
    def __init__(self, llm_connector):
        self.llm_connector = llm_connector

    async def classify_query(self, input_query):
        prompt = self.generate_prompt(input_query)
        response = await self.llm_connector.generate_text(prompt)
        return self.parse_classification(response)

    def generate_prompt(self, input_query):
        return f"""
        You are an AI assistant specializing in query analysis for an advanced image retrieval system. This system can search for images based on visual content, temporal sequences, and extract text information from images.

        **Task**:
        Analyze the given user query and classify it according to the following criteria:

        1. **Temporal Detection**:
           - Determine if the query describes a sequence of events or changes over time.
           - Examples of temporal queries:
             - "A man walks into a store. Then leaves with a bag."
           - Set `"temporal"` to `true` for such queries, `false` otherwise.

        2. **Question Detection**:
           - Identify if the query includes a specific question that requires extracting information from the image.
           - Examples of question queries:
             - "What is the license plate number of the red car?"
             - "How many people are wearing hats in the crowd?"
           - Set `"question"` to `true` for such queries, `false` otherwise.

        3. **Scene Breakdown**:
           - For temporal queries, break down the description into separate scenes.
           - For non-temporal queries, create a single scene.
           - Guidelines for scene descriptions:
             - Use concise, objective language.
             - Focus on visual elements (objects, actions, colors).
             - Avoid subjective interpretations.

        **User Query**:
        "{input_query}"

        **Output Format**:
        Provide a JSON object with the following structure:

        {{
          "temporal": <true/false>,
          "question": <true/false>,
          "number_of_scenes": <integer>,
          "scenes": [
            {{
              "scene": <integer>,
              "description": "<exact word-by-word visual description>",
              "question": <true/false>,
              "specific_question": "<exact question or null>"
            }},
            ...
          ]
        }}

        **Important**:
        - Ensure all boolean values are actual booleans (true/false), not strings.
        - Use proper JSON escaping for string values.
        - For non-temporal queries, set `number_of_scenes` to 1.
        - Include the `specific_question` only in the scene where it's relevant.
        - Provide ONLY the JSON object, no additional text or explanations.
"""

    def parse_classification(self, response):
        try:
            classification = parse_json_response(response)
            return {
                "temporal": classification.get("temporal", False),
                "question": classification.get("question", False),
                "number_of_scenes": classification.get("number_of_scenes", 1),
                "scenes": classification.get("scenes", [
                    {
                        "scene": 1,
                        "description": "",
                        "question": False,
                        "specific_question": None
                    }
                ])
            }
        except Exception as e:
            print(f"Error parsing classification: {e}")
            return {
                "temporal": False,
                "question": False,
                "number_of_scenes": 1,
                "scenes": [
                    {
                        "scene": 1,
                        "description": "",
                        "question": False,
                        "specific_question": None
                    }
                ]
            }
