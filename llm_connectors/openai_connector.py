# openai_connector.py
import openai
import base64
import requests
from .base import LLMConnectorBase

class OpenAIConnector(LLMConnectorBase):
    def __init__(self, api_key, model="gpt-4"):
        openai.api_key = api_key
        self.model = model

    def generate_text(self, prompt, **kwargs):
        messages = kwargs.get('messages', [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ])
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response['choices'][0]['message']['content']

    def analyze_image(self, image_path, prompt, **kwargs):
        # Encode the image to base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Prepare the message content
        message_content = [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]

        messages = kwargs.get('messages', [
            {
                "role": "user",
                "content": message_content
            }
        ])

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get('max_tokens', 300)
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        if response.status_code != 200:
            raise Exception(f"OpenAI API request failed: {response.text}")

        return response.json()['choices'][0]['message']['content']
