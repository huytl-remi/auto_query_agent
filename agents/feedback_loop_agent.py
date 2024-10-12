class FeedbackLoopAgent:
    def __init__(self, llm_connector):
        self.llm_connector = llm_connector

    def handle_feedback(self, input_query, previous_results, attempt):
        prompt = self.generate_prompt(input_query, previous_results, attempt)
        try:
            response = self.llm_connector.call_llm(prompt)
            feedback = self.parse_feedback(response)
            return feedback
        except Exception as e:
            print(f"Error during feedback handling: {e}")
            return None

    def generate_prompt(self, input_query, previous_results, attempt):
        return f"""
You are an advanced assistant for an image retrieval system.

**Situation**:
- **Attempt Number**: {attempt}
- The previous attempt to fulfill the following query was unsuccessful:

**User Query**:
"{input_query}"

**Previous Results**:
{previous_results}

**Task**:
Analyze why the previous attempt failed to produce satisfactory results. Generate a revised strategy to improve the search.

**Instructions**:

1. **Analysis**:
   - Identify potential reasons for the failure (e.g., overly specific criteria, rare scene elements).

2. **Strategy Revision**:
   - Propose adjustments to the search parameters, such as:
     - Broadening the search scope (e.g., expanding temporal windows).
     - Simplifying or rephrasing the scene descriptions.
     - Highlighting different key elements that might yield better results.

3. **Revised Query Generation**:
   - Provide updated scene descriptions and prompts for both the CLIP system and the captioning system.
   - Ensure that the new descriptions maintain the intent of the original query but are adjusted based on your analysis.

**Output Format**:
Provide the revised information as a JSON object with the following structure:

{{
  "scenes": [
    {{
      "scene": <number>,
      "description": "<new scene description>",
      "clip_prompt": "<new CLIP prompt>",
      "caption_prompt": "<new captioning prompt>"
    }},
    ...
  ]
}}

**Important**:
- Maintain consistency with the original query's intent.
- Do not include any additional commentary or explanations.
"""

    def parse_feedback(self, response):
        # Assuming response is already a dict containing revised scenes
        feedback = response
        return feedback
