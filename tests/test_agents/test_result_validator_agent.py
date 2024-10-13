import unittest
from unittest.mock import MagicMock
from agents.result_validator_agent import ResultValidatorAgent

class TestResultValidatorAgent(unittest.TestCase):
    def setUp(self):
        self.mock_llm_connector = MagicMock()
        self.agent = ResultValidatorAgent(self.mock_llm_connector)

    def test_validate_results_success(self):
        image_results = [{'image_path': 'image1.jpg'}]

        # Mock the LLMConnector's analyze_image method
        self.mock_llm_connector.analyze_image.return_value = '''
        {
            "image": "image1.jpg",
            "category": "Exact Match",
            "confidence_score": 95,
            "justification": "The image perfectly matches the query."
        }
        '''

        validated_results = self.agent.validate_results(image_results)
        self.assertEqual(len(validated_results), 1)
        self.assertEqual(validated_results[0]['category'], 'Exact Match')
        self.assertEqual(validated_results[0]['confidence_score'], 95)

    def test_validate_results_failure(self):
        image_results = [{'image_path': 'image1.jpg'}]

        # Simulate an exception in analyze_image
        self.mock_llm_connector.analyze_image.side_effect = Exception("API Error")

        validated_results = self.agent.validate_results(image_results)
        self.assertEqual(len(validated_results), 1)
        self.assertEqual(validated_results[0]['category'], 'No Match')
        self.assertEqual(validated_results[0]['confidence_score'], 0)

if __name__ == '__main__':
    unittest.main()
