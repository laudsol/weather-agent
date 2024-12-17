from dotenv import load_dotenv
import openai
import os
import json
import numpy as np

load_dotenv()
api_key = os.getenv("OPENAI_KEY")
openai.api_key = api_key

def explain_school_closure(location, weather, closure_probability, factors):
    """Use the LLM to generate a user-friendly explanation."""
    prompt = f"""
    You are an assistant helping parents predict school closures based on weather data.
    The user location is {location} and the weather is {weather}
    The model predicts a {closure_probability} likelihood of school closure.
    Here are is a list with the top five factors effecting this estimate: {factors}
    Give the user the probability of school closure. 
    Based on the factors provided and provide the user with insights into why this prediction was made.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[
            {
                "role": "system", 
                "content": "You are an assistant helping users with school closure predictions."
            }, 
            {
                "role": "user", 
                "content": prompt
            }
        ], 
        max_tokens=300, 
        temperature=0.7
    )
    return response['choices'][0]['message']['content']


def classify_user_intent(user_input):
    prompt = f"""
    You are a classification assistant that only deals with school closures due to weather.
    Classify the following user input into one of the categories below.

    Categories:
    - "weather_check": The user is asking if schools will close due to specific weather conditions like snow or storms.
    - "irrelevant": The request is not related to school closures or weather predictions.

    Input: "{user_input}"
    Answer only with the category name.
    """

    assement = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[
            {
                "role": "user", 
                "content": prompt
            }
        ], 
        max_tokens=10, 
        temperature=0
    )
    try:
        return assement['choices'][0]['message']['content']
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return f"Error parsing response: {e}"
    

def classify_additional_factors(additional_input):
    prompt = f"""
    You are a classification assistant that deals with school closures due to weather. 
    The user possesses additional information and wants to know if this information changes the probability of school closure.
    Classify the following user input into categories based on the listed factors. The user may provide information about one or more categories.

    Categories:
    - school_policy: A school policy stating under what conditions the school will close for weather. Policies often specify the number of inches of snowfall or general driving conditions, such as ice and low visibility.
    - prior_closure: The number of days school has already closed for weather-related reasons this year.
    - other: Any other information the user believes is relevant to whether the school will close tomorrow that doesn't fit into the categories above.

    Output Format:
    - Return a JSON-style list of lists, where each sub-list contains two elements:
      - The category name (one of "school_policy", "prior_closure", "other").
      - The corresponding quantified detail (e.g., a number for inches of snow or closure days, or an empty string if not applicable).

    Examples:
    - If the user says: "School policy is to close for 2 inches of snow, and school has closed twice this year."
      - Return: `[["school_policy", 2], ["prior_closure", 2]]`
      
    - If the user says: "I heard from another parent that school might close tomorrow."
      - Return: `[["other", ""]]`
    
    User Input:
    "{additional_input}"
    
    Answer Format: Return only a JSON-style list of lists as described above.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,  # Increased token limit
        temperature=0
    )

    try:
        result = response['choices'][0]['message']['content'].strip()
        parsed_result = json.loads(result)
        
        # check if the data has been returned in the correct format 
        if isinstance(parsed_result, list) and all(isinstance(i, list) and len(i) == 2 for i in parsed_result):
            return parsed_result
        else:
            return "Invalid format returned by the API."

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return f"Error parsing response: {e}"


def validate_location_info(user_input):
    prompt = f"""
    You are a classification assistant that identifies location specificity in user input.

    Categories:
    - specific: A clear and valid location such as a known city, state, or zip code. Example: "New York, 10025".
    - different-specific: More than one valid location provided, like two different zip codes or cities. Example: "New York 10025 and St. Louis 63130".
    - unspecific: Ambiguous location that cannot be mapped to a zip code. Examples: "My town", "My school district", "Springfield" (without a state).
    - none: No location information given at all. Example: "Will schools be closed tomorrow?"

    Classify the following user input into one of these categories:
    Input: "{user_input}"

    Answer only with the category name.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[
            {
                "role": "user", 
                "content": prompt
            }
        ], 
        max_tokens=10, 
        temperature=0
    )
    try: 
        return response['choices'][0]['message']['content']
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return f"Error parsing response: {e}"
    

def extract_location(user_input):
    prompt = f"""
    This prompt contains a location
    Find the zip code associated with this location. If there are multiple zip codes, return the first.
    Input: "{user_input}"

    Answer only the zip code.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[
            {
                "role": "user", 
                "content": prompt
            }
        ], 
        max_tokens=10, 
        temperature=0
    )

    try:
        return response['choices'][0]['message']['content']
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return f"Error parsing response: {e}"