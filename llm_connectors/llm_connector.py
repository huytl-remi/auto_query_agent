# llm_connectors/llm_connector.py
from .base import LLMConnectorBase
from .factory import get_llm_connector

class LLMConnector(LLMConnectorBase):
    def __init__(self, provider_name, api_key=None, model=None):
        self.connector = get_llm_connector(provider_name, api_key, model)

    def generate_text(self, prompt, **kwargs):
        return self.connector.generate_text(prompt, **kwargs)

    def analyze_image(self, image_path, prompt, **kwargs):
        return self.connector.analyze_image(image_path, prompt, **kwargs)
