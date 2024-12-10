from dotenv import load_dotenv
import openai
import os
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
    query_intents = {
        "weather_check": "The user wants to know if schools might close due to weather conditions like snow or storms.",
        "irrelevant": "The request is not about school closures, weather, or related forecasting."
    }

    prompt = f"""
    You are a classification assistant that only deals with school closures due to weather.
    Classify the following user input into one of these categories: {list(query_intents.keys())}.

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
    response = assement['choices'][0]['message']['content']
    response = response if response in query_intents else 'irrelevant' 
    return response

def validate_location_info(user_input):
    location_intents = {
        'specific': 'A clear and valid location such as a known city, state, or zip code. Example: "New York, 10025".',
        'different-specific': 'More than one valid location provided, like two different zip codes or cities. Example: "New York 10025 and St. Louis 63130".',
        'unspecific': 'Ambiguous location that cannot be mapped to a zip code. Examples: "My town", "My school district", "Springfield" (without a state).',
        'none': 'No location information given at all. Example: "Will schools be closed tomorrow?"'
    }

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
    return response['choices'][0]['message']['content']
    

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
    return response['choices'][0]['message']['content']