# agents/query_classifier_agent.py
import json
from utils.json_parser import parse_json_response

class QueryClassifierAgent:
    def __init__(self, llm_connector):
        self.llm_connector = llm_connector

    def classify_query(self, input_query):
        prompt = self.generate_prompt(input_query)
        response = self.llm_connector.generate_text(prompt)
        return self.parse_classification(response)

    def generate_prompt(self, input_query):
        return f"""
You are a highly intelligent assistant specialized in processing and analyzing user queries for an advanced image retrieval system.

**Task**:
Analyze the following user query and perform the following steps:

1. **Temporal Detection**:
   - Determine if the query involves a temporal sequence of events (e.g., actions happening over time).
   - Set `"temporal"` to `true` if it does, or `false` otherwise.

2. **Question Detection**:
   - Determine if the query includes a specific question that requires an answer extracted from the images (e.g., reading text within an image).
   - Set `"question"` to `true` if it does, or `false` otherwise.

3. **Scene Breakdown** (if `"temporal"` is `true`):
   - Break down the query into separate scenes in chronological order.
   - For each scene, provide:
     - `"scene"`: The scene number (starting from 1).
     - `"description"`: A concise description of the scene.
     - `"question"`: `true` or `false` indicating if the scene includes a question.
     - `"specific_question"`: The exact question to be answered in that scene (if any).

**User Query**:
"{input_query}"

**Output Format**:
Provide the analysis as a JSON object with the following structure:

{{
  "temporal": <true/false>,
  "question": <true/false>,
  "number_of_scenes": <number>,
  "scenes": [
    {{
      "scene": <number>,
      "description": "<text>",
      "question": <true/false>,
      "specific_question": "<text or null>"
    }},
    ...
  ]
}}

Ensure that all boolean values are true or false (not strings), and all text fields are properly escaped strings. Do not include any additional commentary.

**Important**:
- Only provide the JSON object in your response.
- Do not add any explanations or extra text.
"""

    def parse_classification(self, response):
        try:
            classification = parse_json_response(response)
            temporal = classification.get("temporal", False)
            question = classification.get("question", False)
            number_of_scenes = classification.get("number_of_scenes", 0)
            scenes = classification.get("scenes", [])
            return {
                "temporal": temporal,
                "question": question,
                "number_of_scenes": number_of_scenes,
                "scenes": scenes
            }
        except Exception as e:
            print(f"Error parsing classification: {e}")
            return {
                "temporal": False,
                "question": False,
                "number_of_scenes": 0,
                "scenes": []
            }
