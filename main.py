# main.py
from agents.query_classifier_agent import QueryClassifierAgent
from agents.prompt_generator_agent import PromptGeneratorAgent
from agents.result_validator_agent import ResultValidatorAgent

class Orchestrator:
    def __init__(self, llm_connector):
        self.llm_connector = llm_connector
        self.query_classifier = QueryClassifierAgent(llm_connector)
        self.prompt_generator = PromptGeneratorAgent(llm_connector)
        self.result_validator = ResultValidatorAgent(llm_connector)
        self.max_attempts = 3  # Prevent infinite loops

    def run_query(self, input_query):
        classification = self.query_classifier.classify_query(input_query)
        attempt = 1
        previous_results = None

        while attempt <= self.max_attempts:
            prompts = self.prompt_generator.generate_prompts(input_query, classification, attempt, previous_results)

            scene_results = []

            for prompt in prompts:
                scene_number = prompt.get('scene', 1)
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
                    'description': classification.get('description', ''),
                    'question': classification.get('question', False),
                    'specific_question': classification.get('specific_question', '')
                }

                scene_results.append(scene_result)

            # Validate results
            validated_results = self.result_validator.validate_results(scene_results)

            # Check if validation produced any exact or near matches
            all_scenes_confirmed = True
            for scene_validation in validated_results:
                if 'error' in scene_validation:
                    all_scenes_confirmed = False
                    break
                summary = scene_validation.get('summary', {})
                exact_matches = summary.get('exact_matches', [])
                near_matches = summary.get('near_matches', [])
                if not exact_matches and not near_matches:
                    all_scenes_confirmed = False
                    break

            if all_scenes_confirmed:
                return validated_results
            else:
                # Prepare for feedback loop
                previous_results = validated_results
                attempt += 1

        # If maximum attempts reached without confirmation
        return None
