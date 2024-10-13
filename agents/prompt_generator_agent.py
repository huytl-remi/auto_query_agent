# agents/prompt_generator_agent.py
from utils.json_parser import parse_json_response

class PromptGeneratorAgent:
    def __init__(self, llm_connector):
        self.llm_connector = llm_connector

    def generate_prompts(self, input_query, classification, attempt=1, previous_results=None):
        prompts = []
        context = f"The overall context is: '{input_query}'"

        if attempt > 1 and previous_results is not None:
            # Modify prompts based on feedback loop
            feedback_prompt = self.generate_feedback_prompt(input_query, previous_results, attempt)
            response = self.llm_connector.generate_text(feedback_prompt)
            feedback = parse_json_response(response)
            classification = feedback  # Assuming feedback provides updated classification

        scenes = classification.get('scenes', [])
        temporal = classification.get('temporal', False)
        if temporal and scenes:
            # Generate prompts for each scene
            for scene in scenes:
                scene_number = scene.get('scene', 1)
                description = scene.get('description', '')
                question = scene.get('question', False)
                specific_question = scene.get('specific_question', None)

                # Generate CLIP prompt
                clip_prompt = f"""
You are an assistant for an image retrieval system using CLIP.

**Task**:
Retrieve frames that match the following scene description:

**Scene {scene_number} Description**:
"{description}"

Focus on capturing all key visual elements mentioned.

**Instructions**:
- Prioritize frames that best match the scene description.
- Do not include irrelevant frames.
- Return the top 500-1000 matching frames.

**Context**:
{context}
"""

                # Generate captioning prompt
                caption_prompt = f"""
You are an assistant for an image captioning system.

**Task**:
Generate detailed captions for frames matching the following scene description:

**Scene {scene_number} Description**:
"{description}"

Include details such as:
- Objects and their attributes
- Actions being performed
- Text within the image (if any)
- Expressions and interactions
"""

                if question:
                    caption_prompt += f"""
**Additional Task**:
Pay special attention to answer the following question:

"{specific_question}"
"""

                caption_prompt += """
**Instructions**:
- Ensure captions are comprehensive and detailed.
- Include any information relevant to the specific question.
"""

                prompts.append({
                    "scene": scene_number,
                    "clip_prompt": clip_prompt.strip(),
                    "caption_prompt": caption_prompt.strip()
                })
        else:
            # Non-temporal query
            description = input_query
            question = classification.get('question', False)
            specific_question = classification.get('specific_question', None)

            # Generate CLIP prompt
            clip_prompt = f"""
You are an assistant for an image retrieval system using CLIP.

**Task**:
Retrieve frames that match the following query:

"{description}"

Focus on capturing all key visual elements mentioned.

**Instructions**:
- Prioritize frames that best match the query description.
- Do not include irrelevant frames.
- Return the top 500-1000 matching frames.
"""

            # Generate captioning prompt
            caption_prompt = f"""
You are an assistant for an image captioning system.

**Task**:
Generate detailed captions for frames matching the following query:

"{description}"

Include details such as:
- Objects and their attributes
- Actions being performed
- Text within the image (if any)
- Expressions and interactions
"""

            if question:
                caption_prompt += f"""
**Additional Task**:
Pay special attention to answer the following question:

"{specific_question}"
"""

            caption_prompt += """
**Instructions**:
- Ensure captions are comprehensive and detailed.
- Include any information relevant to the specific question.
"""

            prompts.append({
                "scene": 1,
                "clip_prompt": clip_prompt.strip(),
                "caption_prompt": caption_prompt.strip()
            })

        return prompts

    def generate_feedback_prompt(self, input_query, previous_results, attempt):
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
   - Provide updated scene descriptions for the new attempt.

**Output Format**:
Provide the revised information as a JSON object with the same structure as the initial classification output:

{{
  "temporal": <true/false>,
  "question": <true/false>,
  "number_of_scenes": <number>,
  "scenes": [
    {{
      "scene": <number>,
      "description": "<new scene description>",
      "question": <true/false>,
      "specific_question": "<text or null>"
    }},
    ...
  ]
}}

**Important**:
- Only include the requested JSON object in your response.
- Do not include any additional commentary or explanations.
"""
