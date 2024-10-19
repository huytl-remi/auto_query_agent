# agents/agent_orchestrator.py
from .query_classifier_agent import QueryClassifierAgent
from .prompt_crafter_agent import PromptCrafterAgent
from .result_validator_agent import ResultValidatorAgent
from services.agent_search_service import AgentSearchService
from utilities.json_parser import parse_json_response
import logging

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    def __init__(self, llm_connector, search_service):
        logger.debug("Initializing AgentOrchestrator")
        self.llm_connector = llm_connector
        self.search_service = search_service
        self.query_classifier = QueryClassifierAgent(llm_connector)
        self.prompt_crafter = PromptCrafterAgent(llm_connector)
        self.result_validator = ResultValidatorAgent(llm_connector)

    async def process_query(self, raw_query, top_k, human_verified_classification=None):
        logger.info(f"Processing query: {raw_query}")
        # Step 1: Query Classification
        if not human_verified_classification:
            classification = await self.query_classifier.classify_query(raw_query)
        else:
            classification = human_verified_classification

        # Step 2: Prompt Crafting
        crafted_prompts = await self.prompt_crafter.craft_prompts(classification)

        # Step 3: Search and Validation
        results = await self.search_and_validate(classification, crafted_prompts, top_k)

        return results

    async def search_and_validate(self, classification, crafted_prompts, top_k):
        logger.debug("Starting search and validation")
        results = []
        for scene_index, scene in enumerate(classification['scenes']):
            scene_results = await self.process_scene(scene, crafted_prompts, scene_index, top_k)
            results.extend(scene_results)

            if classification['temporal'] and scene_index < len(classification['scenes']) - 1:
                exact_match_found = await self.find_next_scene(scene_results, crafted_prompts, scene_index + 1, classification)
                if exact_match_found:
                    # If an exact match is found, stop searching for other alternatives
                    break

        return results

    async def process_scene(self, scene, crafted_prompts, scene_index, top_k):
        logger.debug(f"Processing scene {scene_index}")
        clip_prompt = crafted_prompts['clip_prompts'][scene_index]['prompt']
        caption_prompt = crafted_prompts['caption_prompts'][scene_index]['prompt']

        # Perform search with user-provided top_k
        search_results = await self.search_service.agent_search(clip_prompt, caption_prompt, top_k)

        # Validate results
        validated_results = await self.result_validator.validate_results(search_results, crafted_prompts)

        return validated_results

    async def find_next_scene(self, current_scene_results, crafted_prompts, next_scene_index, classification):
        logger.debug(f"Finding next scene. Current index: {next_scene_index - 1}")
        exact_match_found = False
        matching_frame = None

        # Sort current_scene_results by match category and confidence
        sorted_results = sorted(
            current_scene_results,
            key=lambda x: ('2' if x['match_assessment']['category'] == 'Near Match' else '1', -x['match_assessment']['confidence'])
        )

        for result in sorted_results:
            if result['match_assessment']['category'] in ["Exact Match", "Near Match"]:
                next_frames = await self.search_service.get_next_frames(result['image_path'], 3)
                next_scene_prompt = crafted_prompts['clip_prompts'][next_scene_index]['prompt']
                validated_next_frames = await self.result_validator.validate_results(next_frames, {'clip_prompts': [{'prompt': next_scene_prompt}]})

                # Check if any of the validated next frames is an exact match
                for next_frame in validated_next_frames:
                    if next_frame['match_assessment']['category'] == "Exact Match":
                        exact_match_found = True
                        matching_frame = next_frame
                        result['next_scene'] = matching_frame
                        break

                if exact_match_found:
                    break

        if exact_match_found and matching_frame:
            # If there are more scenes, continue the search
            if next_scene_index + 1 < len(classification['scenes']):
                next_exact_match = await self.find_next_scene([matching_frame], crafted_prompts, next_scene_index + 1, classification)
                return next_exact_match
            else:
                return True

        return False
