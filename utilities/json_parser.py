import json
import re

def parse_json_response(response):
    if isinstance(response, dict):
        return response

    if not isinstance(response, str):
        raise ValueError("Input must be a string or dictionary")

    # Remove markdown code blocks if present
    cleaned_response = re.sub(r'```json\s*|\s*```', '', response).strip()

    # If that didn't work, try to find JSON within the string
    if not cleaned_response.startswith('{'):
        json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
        if json_match:
            cleaned_response = json_match.group()

    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")
