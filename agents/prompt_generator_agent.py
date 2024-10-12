class PromptGeneratorAgent:
    def __init__(self, llm_connector):
        self.llm_connector = llm_connector

    def generate_prompts(self, input_query, classification):
        prompts = []
        context = f"The overall context is: '{input_query}'"

        scenes = classification.get('scenes', [])
        if classification.get('temporal', False) and scenes:
            # Generate prompts for each scene
            for scene in scenes:
                scene_number = scene.get('scene')
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
