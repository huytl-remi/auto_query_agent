# tests/test_services/test_validation_service.py
import unittest
from unittest.mock import MagicMock
from services.validation_service import run_ai_validator

class TestValidationService(unittest.TestCase):
    def setUp(self):
        self.image_paths = ['image1.jpg', 'image2.jpg']
        self.default_llm_provider = 'openai'
        self.config = MagicMock()
        self.config.DEFAULT_LLM_PROVIDER = 'openai'
        self.config.get_api_key.return_value = 'test-api-key'

    def test_run_ai_validator_success(self):
        # Mock LLMConnector and ResultValidatorAgent
        with unittest.mock.patch('llm_connectors.llm_connector.LLMConnector') as MockLLMConnector, \
             unittest.mock.patch('agents.result_validator_agent.ResultValidatorAgent') as MockValidatorAgent:

            mock_connector = MockLLMConnector.return_value
            mock_agent = MockValidatorAgent.return_value
            mock_agent.validate_results.return_value = [
                {
                    'image': 'image1.jpg',
                    'category': 'Exact Match',
                    'confidence_score': 95,
                    'justification': 'Perfectly matches the query.'
                },
                {
                    'image': 'image2.jpg',
                    'category': 'Near Match',
                    'confidence_score': 85,
                    'justification': 'Closely matches with minor differences.'
                }
            ]

            # Execute
            validated_results = run_ai_validator(
                image_paths=self.image_paths,
                default_llm_provider=self.default_llm_provider,
                config=self.config
            )

            # Verify
            self.assertEqual(len(validated_results), 2)
            self.assertEqual(validated_results[0]['category'], 'Exact Match')
            self.assertEqual(validated_results[1]['category'], 'Near Match')
            MockLLMConnector.assert_called_once_with(
                provider_name=self.default_llm_provider,
                api_key='test-api-key'
            )
            MockValidatorAgent.assert_called_once_with(mock_connector)
            mock_agent.validate_results.assert_called_once_with(
                [{'image_path': 'image1.jpg'}, {'image_path': 'image2.jpg'}]
            )

    def test_run_ai_validator_failure(self):
        # Mock LLMConnector and ResultValidatorAgent to raise exception
        with unittest.mock.patch('llm_connectors.llm_connector.LLMConnector') as MockLLMConnector, \
             unittest.mock.patch('agents.result_validator_agent.ResultValidatorAgent') as MockValidatorAgent:

            mock_connector = MockLLMConnector.return_value
            mock_agent = MockValidatorAgent.return_value
            mock_agent.validate_results.side_effect = Exception("Validation Error")

            # Execute and expect exception
            with self.assertRaises(Exception) as context:
                run_ai_validator(
                    image_paths=self.image_paths,
                    default_llm_provider=self.default_llm_provider,
                    config=self.config
                )

            self.assertTrue('Validation Error' in str(context.exception))
            MockLLMConnector.assert_called_once_with(
                provider_name=self.default_llm_provider,
                api_key='test-api-key'
            )
            MockValidatorAgent.assert_called_once_with(mock_connector)
            mock_agent.validate_results.assert_called_once_with(
                [{'image_path': 'image1.jpg'}, {'image_path': 'image2.jpg'}]
            )

if __name__ == '__main__':
    unittest.main()
