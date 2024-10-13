# anthropic_connector.py
import anthropic
import base64
import httpx
from .base import LLMConnectorBase

class AnthropicConnector(LLMConnectorBase):
    def __init__(self, api_key, model="claude-2"):
        self.client = anthropic.Client(api_key=api_key)
        self.model = model

    def generate_text(self, prompt, **kwargs):
        response = self.client.completions.create(
            model=self.model,
            max_tokens_to_sample=kwargs.get('max_tokens', 1000),
            prompt=prompt,
            **kwargs
        )
        return response['completion']

    def analyze_image(self, image_path, prompt, **kwargs):
        # Encode image to base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

        image_media_type = "image/jpeg"  # Adjust as needed

        # Build the message content
        message_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_media_type,
                    "data": image_data,
                },
            },
            {
                "type": "text",
                "text": prompt
            }
        ]

        response = self.client.completions.create(
            model=self.model,
            max_tokens_to_sample=kwargs.get('max_tokens', 1024),
            prompt=message_content,
            **kwargs
        )
        return response['completion']
