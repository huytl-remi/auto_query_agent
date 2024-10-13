# agents/result_validator_agent.py
import json
from utils.json_parser import parse_json_response

class ResultValidatorAgent:
    def __init__(self, llm_connector):
        self.llm_connector = llm_connector

    def validate_results(self, scene_results):
        validated_results = []
        for scene_result in scene_results:
            scene_number = scene_result.get('scene', 1)
            results = scene_result.get('results', [])
            description = scene_result.get('description', '')
            question = scene_result.get('question', False)
            specific_question = scene_result.get('specific_question', None)

            prompt = self.generate_prompt(scene_number, results, description, question, specific_question)

            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    response = self.llm_connector.generate_text(prompt)
                    validation = self.parse_validation(response)
                    validation['scene'] = scene_number
                    validated_results.append(validation)
                    break  # Break out of retry loop if successful
                except Exception as e:
                    print(f"Error during validation for scene {scene_number}, attempt {attempt + 1}: {e}")
                    if attempt == max_retries:
                        print(f"Max retries reached for scene {scene_number}.")
                        # Append an error result
                        validated_results.append({
                            'scene': scene_number,
                            'error': str(e),
                            'results': [],
                            'summary': {
                                'exact_matches': [],
                                'near_matches': [],
                                'weak_matches': [],
                                'no_matches': results  # Assume all results are no match
                            },
                            'question_answer': None
                        })

        return validated_results

    def generate_prompt(self, scene_number, results, description, question, specific_question):
        prompt = f"""
You are an expert image analyst.

**Task**:
Validate and categorize a list of image frames based on how well they match a given scene description.

**Scene {scene_number} Description**:
"{description}"

**Image Frames**:
{results}

**Instructions**:
For each image frame:

1. **Compare** the image to the scene description.
2. **Provide** a **confidence score** between 0 and 100 indicating how well the image matches the description.
3. **Categorize** each image into one of the following categories:
- **Exact Match**: The image perfectly matches all aspects of the scene description.
- **Near Match**: The image closely matches the description but may have minor differences.
- **Weak Match**: The image has some elements matching the description but significant details are missing.
- **No Match**: The image does not match the scene description.
4. **Justify** your categorization with a brief explanation (one or two sentences).

**Output Format**:
Provide a JSON object with the following structure:

{{
  "results": [
    {{
      "image": "<image_filename>",
      "category": "<Exact Match/Near Match/Weak Match/No Match>",
      "confidence_score": <number between 0 and 100>,
      "justification": "<brief explanation>"
    }},
    ...
  ],
  "summary": {{
    "exact_matches": [<list of image filenames>],
    "near_matches": [<list of image filenames>],
    "weak_matches": [<list of image filenames>],
    "no_matches": [<list of image filenames>]
  }}
}}
"""

        if question:
            prompt += f"""
**Additional Task**:
For images categorized as **Exact Match** or **Near Match**, attempt to answer the following question:

"{specific_question}"

**Include** the answer in the JSON output under the key `"answer"`.

**Final Output Format**:
{{
  "results": [
    {{
      "image": "<image_filename>",
      "category": "<category>",
      "confidence_score": <number>,
      "justification": "<text>",
      "answer": "<answer_text or null>"
    }},
    ...
  ],
  "summary": {{
    "exact_matches": [<list of image filenames>],
    "near_matches": [<list of image filenames>],
    "weak_matches": [<list of image filenames>],
    "no_matches": [<list of image filenames>]
  }},
  "question_answer": "<final answer to the question or null>"
}}
"""

        prompt += """
**Important**:
- Only include the requested JSON object in your response.
- Ensure all numeric values are numbers (not strings).
- Do not add any additional commentary or explanations.
"""

        return prompt

    def parse_validation(self, response):
        try:
            validation = parse_json_response(response)

            # Ensure that the 'summary' and 'results' fields exist
            if "summary" not in validation or "results" not in validation:
                raise KeyError("Missing required fields 'summary' or 'results'.")

            # Ensure that 'results' is a list
            if not isinstance(validation['results'], list):
                raise TypeError("'results' should be a list.")

            # Iterate through the 'results' and check each entry
            for result in validation['results']:
                # Ensure required keys are present
                if not all(key in result for key in ['image', 'category', 'confidence_score', 'justification']):
                    raise KeyError("Missing required keys in one of the result items.")

                # Validate types
                if not isinstance(result['image'], str):
                    raise TypeError(f"'image' should be a string. Got {type(result['image'])}.")
                if not isinstance(result['category'], str):
                    raise TypeError(f"'category' should be a string. Got {type(result['category'])}.")
                if not isinstance(result['confidence_score'], (int, float)):
                    raise TypeError(f"'confidence_score' should be a number. Got {type(result['confidence_score'])}.")
                if not isinstance(result['justification'], str):
                    raise TypeError(f"'justification' should be a string. Got {type(result['justification'])}.")

                # Optionally check for the 'answer' field if it's a question-based query
                if 'answer' in result and not isinstance(result['answer'], (str, type(None))):
                    raise TypeError(f"'answer' should be a string or None. Got {type(result['answer'])}.")

            return validation
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error parsing validation: {e}")
            raise ValueError("Failed to parse or validate the LLM response correctly.")
