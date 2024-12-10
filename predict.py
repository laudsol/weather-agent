import pandas as pd
import joblib
import numpy as np
from llm_factory import explain_school_closure, classify_user_intent, validate_location_info, extract_location

class SchoolClosureAgent:
    def __init__(self, model_path, feature_names):
        self.model = joblib.load(model_path)
        self.feature_names = feature_names

    def fetch_weather_data(self, location):
        """Simulate fetching weather data for the given location."""
        print(f"\nFetching weather data for {location}...\n")
        # Hardcoded weather data (simulate API response)
        weather_data = {
            'temp': 23,           # Average temperature in Fahrenheit
            'visibility': 5,      # Visibility in miles
            'dew_point': 22,      # Dew point in Fahrenheit
            'feels_like': 20,     # Feels like temperature in Fahrenheit
            'temp_min': 22,       # Minimum temperature in Fahrenheit
            'temp_max': 26,       # Maximum temperature in Fahrenheit
            'pressure': 1020,     # Atmospheric pressure in hPa
            'humidity': 85,       # Humidity percentage
            'wind_speed': 18,     # Wind speed in mph
            'wind_gust': 20,      # Wind gust in mph
            'wind_deg': 70,      # Wind direction in degrees
            'rain_1h': 0,         # Rain volume for the last 1 hour in mm
            'rain_3h': 0,         # Rain volume for the last 3 hours in mm
            'snow_1h': 4,         # Snow volume for the last 1 hour in mm
            'snow_3h': 7,         # Snow volume for the last 3 hours in mm
            'clouds_all': 100,     # Cloudiness percentage
            'weather_id': 602,    # Weather condition code
            'is_weekend': 0,      # 1 if weekend, else 0
            'is_holiday': 0,      # 1 if holiday, else 0
            'is_covid_period': 0  # 1 if during COVID period, else 0
        }
        return weather_data

    def predict_closure(self, weather_features):
        """Predict the probability of school closure and explain feature importance."""
        # Convert weather features into DataFrame
        X_input = pd.DataFrame([weather_features])

        probability = self.model.predict_proba(X_input)[0][1]
        importances = self.model.feature_importances_
        feature_importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        return probability, feature_importances

    def generate_response(self, probability, feature_importances):
        """Generate a response to the user."""
        top_features = feature_importances.head(5)

        explanation = "The most influential factors contributing to this prediction are:\n"
        for index, row in top_features.iterrows():
            feature_name = row['feature'].replace('_', ' ').title()
            explanation += f"- **{feature_name}** (importance score: {row['importance']:.2f})\n"

        response = f"""
        There is a **{probability * 100:.1f}% chance** that schools will be closed tomorrow.
        {explanation}
        """

        return response
    
    def get_valid_location(self, location_validator, user_input):
            if (location_validator == 'specific'):
                print('specific')
                return extract_location(user_input)
            elif (location_validator == 'different-specfiic'):
                location_validator = input("Sorry, I'm having trouble determinig the location you are inquiring about. Can you please clarify? (e.g. enter a city name or zip code)").strip()
                return self.get_valid_location(location_validator)
            elif (location_validator == 'unspecific'):
                location_validator = input("Please specify the location you are inquiring about: (e.g. enter a city name or zip code)").strip()
                return self.get_valid_location(location_validator)
            else: 
                location_validator = input("Please specify the location you are inquiring about: (e.g. enter a city name or zip code)").strip()
                return self.get_valid_location(location_validator)

    def handle_interaction(self):
        print("Welcome to the School Closure Prediction Agent!")
        print("I can help you predict school closures based on weather conditions.\n")

        user_input = input("How can I assist you today? (e.g., 'Will schools close tomorrow in Boston?'): ").strip()

        while True:
            input_assessment = classify_user_intent(user_input)

            if "exit" in user_input.lower():
                print("Thank you for using the School Closure Prediction Agent. Goodbye!")
                break
            elif input_assessment == 'irrelevant':
                print("I'm sorry that's not something I can help with. Please try another request:")
            else: 

                location_validator = validate_location_info(user_input)
                location = self.get_valid_location(location_validator, user_input)
                
                weather_data = self.fetch_weather_data(location)
                closure_probability, factors = self.predict_closure(weather_data)
                response = explain_school_closure(location, weather_data, closure_probability, factors.head(5))
                print(response)
                
            user_input = input("Can I help you with anything else?").strip()

if __name__ == "__main__":
    feature_names = [
        "temp", "visibility", "dew_point", "feels_like", "temp_min", "temp_max",
        "pressure", "humidity", "wind_speed", "wind_gust", "wind_deg",
        "rain_1h", "rain_3h", "snow_1h", "snow_3h", "clouds_all", "weather_id",
        "is_weekend", "is_holiday", "is_covid_period"
    ]

    agent = SchoolClosureAgent(model_path='./models/school_closure_model.pkl', feature_names=feature_names)
    agent.handle_interaction()
