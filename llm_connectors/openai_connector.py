# openai_connector.py
import openai
import base64
import aiohttp
import asyncio
import aiofiles
from .base import LLMConnectorBase

class OpenAIConnector(LLMConnectorBase):
    def __init__(self, api_key, model="gpt-4"):
        openai.api_key = api_key
        self.model = model

    async def generate_text(self, prompt, **kwargs):
        messages = kwargs.get('messages', [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ])

        async with aiohttp.ClientSession() as session:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai.api_key}"
            }
            payload = {
                "model": self.model,
                "messages": messages,
                **kwargs
            }
            async with session.post("https://api.openai.com/v1/chat/completions",
                                    headers=headers, json=payload) as resp:
                response = await resp.json()
                if resp.status != 200:
                    raise Exception(f"OpenAI API request failed: {response}")
                return response['choices'][0]['message']['content']

    async def analyze_image(self, image_path, prompt, **kwargs):
        # Asynchronously read the image file
        async with aiofiles.open(image_path, "rb") as image_file:
            image_data = await image_file.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')

        # Prepare the message content
        message_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]

        # Asynchronously call the OpenAI API
        async with aiohttp.ClientSession() as session:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai.api_key}"
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": message_content}],
                "max_tokens": kwargs.get('max_tokens', 300)
            }
            async with session.post("https://api.openai.com/v1/chat/completions",
                                    headers=headers, json=payload) as resp:
                response = await resp.json()
                if resp.status != 200:
                    raise Exception(f"OpenAI API request failed: {response}")
                return response['choices'][0]['message']['content']
