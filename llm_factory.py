from dotenv import load_dotenv
import openai
import os
import numpy as np

load_dotenv()
api_key = os.getenv("OPENAI_KEY")
openai.api_key = api_key

def generate_llm_response(location, weather, closure_probability, factors):
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