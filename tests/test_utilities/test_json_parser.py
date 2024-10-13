import unittest
from utilities.json_parser import parse_json_response

class TestJSONParser(unittest.TestCase):
    def test_parse_valid_json(self):
        response = '''
        Some text before JSON.
        {
            "key": "value",
            "number": 123
        }
        Some text after JSON.
        '''
        result = parse_json_response(response)
        self.assertEqual(result['key'], 'value')
        self.assertEqual(result['number'], 123)

    def test_parse_no_json(self):
        response = 'No JSON here.'
        with self.assertRaises(ValueError):
            parse_json_response(response)

    def test_parse_invalid_json(self):
        response = '{invalid json}'
        with self.assertRaises(ValueError):
            parse_json_response(response)

if __name__ == '__main__':
    unittest.main()
