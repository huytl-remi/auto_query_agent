# llm_connector_base.py
from abc import ABC, abstractmethod

class LLMConnectorBase(ABC):
    @abstractmethod
    def generate_text(self, prompt, **kwargs):
        pass

    @abstractmethod
    def analyze_image(self, image_path, prompt, **kwargs):
        pass
