from agents.query_classifier_agent import QueryClassifierAgent
from agents.prompt_generator_agent import PromptGeneratorAgent
from agents.result_validator_agent import ResultValidatorAgent
from agents.feedback_loop_agent import FeedbackLoopAgent

from llm_connectors.llm_connector import LLMConnector

import streamlit as st

# Configuration
PROVIDER_NAME = "openai"  # or "anthropic", "gemini"
API_KEY = "YOUR_API_KEY"
MODEL_NAME = None  # Use default if None

# Instantiate the LLMConnector
llm_connector = LLMConnector(provider_name=PROVIDER_NAME, api_key=API_KEY, model=MODEL_NAME)

class Orchestrator:
    def __init__(self, llm_connector):
        self.llm_connector = llm_connector
        self.query_classifier = QueryClassifierAgent(llm_connector)
        self.prompt_generator = PromptGeneratorAgent(llm_connector)
        self.result_validator = ResultValidatorAgent(llm_connector)
        self.feedback_loop = FeedbackLoopAgent(llm_connector)
        self.max_attempts = 3  # Prevent infinite loops

    def run_query(self, input_query):
        classification = self.query_classifier.classify_query(input_query)
        attempt = 1

        while attempt <= self.max_attempts:
            st.write(f"### Attempt {attempt}")
            prompts = self.prompt_generator.generate_prompts(input_query, classification)

            scene_results = []

            for prompt in prompts:
                scene_number = prompt.get('scene')
                clip_prompt = prompt.get('clip_prompt')
                caption_prompt = prompt.get('caption_prompt')

                # Retrieve results from CLIP and captioning systems
                # For simplicity, we'll mock the results
                clip_results = [f"scene{scene_number}_image1.jpg", f"scene{scene_number}_image2.jpg"]
                caption_results = [f"scene{scene_number}_image3.jpg", f"scene{scene_number}_image4.jpg"]

                # Remove duplicates
                combined_results = list(set(clip_results + caption_results))

                # Build scene result
                scene_result = {
                    'scene': scene_number,
                    'results': combined_results,
                    'description': '',
                    'question': False,
                    'specific_question': ''
                }

                # Get scene details from classification
                for scene in classification.get('scenes', []):
                    if scene.get('scene') == scene_number:
                        scene_result['description'] = scene.get('description', '')
                        scene_result['question'] = scene.get('question', False)
                        scene_result['specific_question'] = scene.get('specific_question', '')
                        break

                scene_results.append(scene_result)

            # Validate results
            validated_results = self.result_validator.validate_results(scene_results)

            # Human review and confirmation using Streamlit
            confirmed_scenes = []
            for scene_validation in validated_results:
                scene_number = scene_validation.get('scene')
                st.write(f"#### Scene {scene_number} Results")

                if 'error' in scene_validation:
                    st.write(f"Error in validation: {scene_validation['error']}")
                    continue

                # Display results with confidence scores and justifications
                results = scene_validation.get('results', [])
                if results:
                    for result in results:
                        image = result.get('image')
                        category = result.get('category')
                        confidence_score = result.get('confidence_score')
                        justification = result.get('justification')
                        st.write(f"- **Image**: {image}")
                        st.write(f"  - **Category**: {category}")
                        st.write(f"  - **Confidence Score**: {confidence_score}")
                        st.write(f"  - **Justification**: {justification}")
                        if 'answer' in result and result['answer']:
                            st.write(f"  - **Answer**: {result['answer']}")

                    # Mock human confirmation
                    confirmed = st.checkbox(f"Confirm Scene {scene_number}?", key=f"scene_{scene_number}_confirm")
                    if confirmed:
                        confirmed_scenes.append(scene_number)
                else:
                    st.write("No results after validation.")

                # Display question answer if available
                question_answer = scene_validation.get('question_answer')
                if question_answer:
                    st.write(f"**Answer to the question**: {question_answer}")

            # Check if all scenes are confirmed
            if len(confirmed_scenes) == len(prompts):
                st.write("All scenes confirmed.")
                return validated_results

            else:
                st.write("Some scenes not confirmed. Triggering feedback loop.")
                feedback = self.feedback_loop.handle_feedback(input_query, validated_results, attempt)
                if feedback:
                    # Update classification and prompts based on feedback
                    classification = feedback  # Assuming feedback provides updated classification
                    attempt += 1
                else:
                    st.write("Feedback loop failed to generate new prompts.")
                    break

        st.write("Maximum attempts reached. Query unsuccessful.")
        return None
