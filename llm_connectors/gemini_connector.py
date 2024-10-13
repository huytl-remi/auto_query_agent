# gemini_connector.py
import google.generativeai as genai
from .base import LLMConnectorBase

class GeminiConnector(LLMConnectorBase):
    def __init__(self, api_key, model="gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = model

    def generate_text(self, prompt, **kwargs):
        response = genai.generate_text(
            model=self.model,
            prompt=prompt,
            **kwargs
        )
        return response.result

    def analyze_image(self, image_path, prompt, **kwargs):
        # Upload the image file
        uploaded_file = genai.upload_file(image_path)

        # Build the content
        content = [
            uploaded_file,
            "\n\n",
            prompt
        ]

        response = genai.generate_text(
            model=self.model,
            prompt=content,
            **kwargs
        )
        return response.result
