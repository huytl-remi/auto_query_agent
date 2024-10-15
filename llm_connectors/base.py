# llm_connectors/base.py
from abc import ABC, abstractmethod

class LLMConnectorBase(ABC):
    @abstractmethod
    async def generate_text(self, prompt, **kwargs):
        pass

    @abstractmethod
    async def analyze_image(self, image_path, prompt, **kwargs):
        pass
