import unittest
from llm_factory import explain_school_closure, classify_user_intent, validate_location_info, extract_location

class TestOpenAICalls(unittest.TestCase):
    
    def test_explain_school_closure(self):
        input = ""
        response = explain_school_closure(input)
        self.assertEqual(response, 'weather_check')
    
    def test_classify_user_intent(self):
        test_case = 1
        test_inputs = [
            'Will schools be closed tomrrow in St Louis MO?',
            'Can I use you to check if schools will be closed tomorrow?',
            'What is the meaning of life?',
        ]
        expected_results = [
            'weather_check',
            'weather_check',
            'irrelevant',
        ]
        input = test_inputs[test_case]
        response = classify_user_intent(input)
        expected = expected_results[test_case]

        self.assertEqual(response, expected)

    def test_validate_location_info(self):
        test_case = 1
        test_inputs = [
            'will schools be closed tomorrow in Los Angeles, California',
            'will schools be closed tomorrow in Springfield, Chicago, Maryland',
            'will schools be closed tomorrow in My town',
            'will schools be closed tomorrow'
        ]
        expected_results = [
            'specific',
            'different-specific',
            'unspecific',
            'none'
        ]
        input = test_inputs[test_case]
        response = validate_location_info(input)
        expected = expected_results[test_case]

        self.assertEqual(response, expected)

    def test_extract_location(self):
        test_case = 0
        test_inputs = [
            'will schools be closed tomorrow in Los Angeles, California',
            'will schools be closed tomorrow in Springfield, Missouri',
            'will schools be closed tomorrow in 63130'
        ]
        expected_results = [
            '90001'
            '65619',
            '63130',
        ]
        input = test_inputs[test_case]
        response = validate_location_info(input)
        expected = expected_results[test_case]

        self.assertEqual(response, expected)


if __name__ == '__main__':
    unittest.main()
