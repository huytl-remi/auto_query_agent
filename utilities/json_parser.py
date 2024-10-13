# utils/json_parser.py
import json
import re

def parse_json_response(response):
    """
    Extracts and parses the JSON object from the LLM response,
    handling different formats and providers.
    """
    try:
        # Use regex to find the first JSON object in the response
        json_pattern = re.compile(r'\{.*\}', re.DOTALL)
        match = json_pattern.search(response)
        if match:
            json_str = match.group()
            return json.loads(json_str)
        else:
            raise ValueError("No JSON object found in the response.")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing error: {e}")
