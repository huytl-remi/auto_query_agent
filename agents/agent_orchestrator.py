# agents/agent_orchestrator.py

from .query_classifier_agent import QueryClassifierAgent
from .prompt_crafter_agent import PromptCrafterAgent
from .result_validator_agent import ResultValidatorAgent
from services.agent_search_service import AgentSearchService
from utilities.json_parser import parse_json_response

class AgentOrchestrator:
    def __init__(self, llm_connector, search_service):
        self.llm_connector = llm_connector
        self.search_service = search_service
        self.query_classifier = QueryClassifierAgent(llm_connector)
        self.prompt_crafter = PromptCrafterAgent(llm_connector)
        self.result_validator = ResultValidatorAgent(llm_connector)

    async def process_query(self, raw_query, human_verified_classification=None):
        # Step 1: Query Classification
        if not human_verified_classification:
            classification = self.query_classifier.classify_query(raw_query)
        else:
            classification = human_verified_classification

        # Step 2: Prompt Crafting
        crafted_prompts = self.prompt_crafter.craft_prompts(classification)

        # Step 3: Search and Validation
        results = await self.search_and_validate(classification, crafted_prompts)

        return results

    async def search_and_validate(self, classification, crafted_prompts):
        results = []
        for scene_index, scene in enumerate(classification['scenes']):
            scene_results = await self.process_scene(scene, crafted_prompts, scene_index)
            results.extend(scene_results)

            if classification['temporal'] and scene_index < len(classification['scenes']) - 1:
                next_scene_results = await self.find_next_scene(scene_results, crafted_prompts, scene_index + 1)
                results.extend(next_scene_results)

        return results

    async def process_scene(self, scene, crafted_prompts, scene_index):
        clip_prompt = crafted_prompts['clip_prompts'][scene_index]['prompt']
        caption_prompt = crafted_prompts['caption_prompts'][scene_index]['prompt']

        # Perform high top_p search
        search_results = await self.search_service.agent_search(clip_prompt, caption_prompt, top_p=500)

        # Validate results
        validated_results = await self.result_validator.validate_results(search_results, crafted_prompts)

        return validated_results

    async def find_next_scene(self, current_scene_results, crafted_prompts, next_scene_index):
        next_scene_results = []
        for result in current_scene_results:
            if result['match_assessment']['category'] == "Exact Match":
                next_frames = await self.search_service.get_next_frames(result['image_path'], 10)
                next_scene_prompt = crafted_prompts['clip_prompts'][next_scene_index]['prompt']
                validated_next_frames = await self.result_validator.validate_results(next_frames, {'clip_prompts': [{'prompt': next_scene_prompt}]})
                next_scene_results.extend(validated_next_frames)

        return next_scene_results

    def extract_answer(self, results, question):
        for result in results:
            if result['question_answer']['question'] == question:
                return result['question_answer']['answer']
        return None
