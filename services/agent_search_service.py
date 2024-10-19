# services/agent_search_service.py

import asyncio
import os
from utilities.model_utils import search_image_by_text, search_image_by_text_with_captioning, get_image_paths, search_images_by_ocr
from data_loaders.metadata_loader import load_ocr_data, load_object_data, load_object_count_data
from utilities.video_utils import get_temporal_frames

class AgentSearchService:
    def __init__(self, model, index, id2img_fps):
        self.model = model
        self.index = index
        self.id2img_fps = id2img_fps

    async def agent_search(self, clip_prompt, caption_prompt, top_k):
        clip_results = await self.clip_search(clip_prompt, top_k)
        caption_results = await self.caption_search(caption_prompt, top_k)

        combined_results = self.combine_results(clip_results, caption_results)
        return combined_results

    async def clip_search(self, prompt, top_k):
        image_indices, distances = search_image_by_text(self.model, self.index, prompt, top_k)
        image_paths = get_image_paths(image_indices, self.id2img_fps)
        return [{'image_path': path, 'distance': dist} for path, dist in zip(image_paths, distances[0])]

    async def caption_search(self, prompt, top_k):
        image_paths = search_image_by_text_with_captioning(prompt, top_k)
        return [{'image_path': path} for path in image_paths]

    def combine_results(self, clip_results, caption_results):
        combined = {}
        for result in clip_results + caption_results:
            if result['image_path'] not in combined:
                combined[result['image_path']] = result
            elif 'distance' in result:
                combined[result['image_path']]['distance'] = result['distance']
        return list(combined.values())

    async def get_next_frames(self, image_path, num_frames):
        surrounding_frames = get_temporal_frames(image_path, min_distance=1, temporal_range=num_frames)
        return [{'image_path': frame} for frame in surrounding_frames]

    async def ocr_search(self, text_query, top_p):
        image_paths = search_images_by_ocr(text_query, top_p)
        return [{'image_path': path} for path in image_paths]

    async def load_metadata(self, image_paths):
        ocr_data, _ = load_ocr_data(image_paths)
        object_data, _ = load_object_data(image_paths)
        count_data, _ = load_object_count_data(image_paths)

        metadata = {}
        for path in image_paths:
            metadata[path] = {
                'ocr': ocr_data.get(path, []),
                'objects': object_data.get(path, []),
                'counts': count_data.get(path, [])
            }
        return metadata
