# anthropic_connector.py
import base64
import httpx
import aiofiles
from .base import LLMConnectorBase

class AnthropicConnector(LLMConnectorBase):
    def __init__(self, api_key, model="claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1"

    async def generate_text(self, prompt, **kwargs):
        async with httpx.AsyncClient() as client:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            payload = {
                "model": self.model,
                "max_tokens": kwargs.get('max_tokens', 1000),
                "messages": [{"role": "user", "content": prompt}]
            }
            response = await client.post(f"{self.base_url}/messages", headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result['content'][0]['text']

    async def analyze_image(self, image_path, prompt, **kwargs):
        async with httpx.AsyncClient() as client:
            async with aiofiles.open(image_path, "rb") as image_file:
                image_data = await image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')

            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            payload = {
                "model": self.model,
                "max_tokens": kwargs.get('max_tokens', 1024),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
            response = await client.post(f"{self.base_url}/messages", headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result['content'][0]['text']