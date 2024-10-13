import unittest
from unittest.mock import patch, MagicMock
from llm_connectors.openai_connector import OpenAIConnector

class TestOpenAIConnector(unittest.TestCase):
    def setUp(self):
        self.api_key = 'test-api-key'
        self.model = 'gpt-4'
        self.connector = OpenAIConnector(api_key=self.api_key, model=self.model)

    @patch('openai.ChatCompletion.create')
    def test_generate_text(self, mock_create):
        # Mock the OpenAI API response
        mock_create.return_value = {
            'choices': [{'message': {'content': 'Test response'}}]
        }

        prompt = 'Hello, world!'
        response = self.connector.generate_text(prompt)
        self.assertEqual(response, 'Test response')
        mock_create.assert_called_once()

    @patch('requests.post')
    def test_analyze_image(self, mock_post):
        # Mock the API response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            'choices': [{'message': {'content': 'Image analysis result'}}]
        }

        image_path = 'test_image.jpg'
        prompt = 'Analyze this image.'

        with patch('builtins.open', unittest.mock.mock_open(read_data=b'test data')):
            response = self.connector.analyze_image(image_path, prompt)
            self.assertEqual(response, 'Image analysis result')
            mock_post.assert_called_once()

if __name__ == '__main__':
    unittest.main()
