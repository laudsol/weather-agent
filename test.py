import unittest
from llm_factory import explain_school_closure, classify_user_intent, validate_location_info, extract_location, classify_additional_factors

# to test: python3 -m unittest test.TestOpenAICalls.test_function_name

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

    def test_classify_additional_factors(self):
        test_case = 2
        test_inputs = [
            'School policy says that school will close if there are two or more inches of snow. School hasn\'nt closed yet this year.',
            'School has been closed unexpectedly for four days this year but only three were related to weather. One of the teachers told me she thought school wouldn\'t close tomorrow.',
            'Administration sent an email remining us school only closes if there is four inches of snow'
        ]
        expected_results = [
            [['school_policy', 2], ['prior_closure', 0]],
            [['prior_closure', 3], ['other', '']],
            [['school_policy', 4]]
        ]
        input = test_inputs[test_case]
        response = classify_additional_factors(input)
        expected = expected_results[test_case]

        self.assertEqual(response, expected)

if __name__ == '__main__':
    unittest.main()

