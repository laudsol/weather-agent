import pandas as pd
import joblib
import numpy as np
from llm_factory import explain_school_closure, classify_user_intent, validate_location_info, extract_location, classify_additional_factors
from model_mapper import zip_to_model

class SchoolClosureAgent:
    def __init__(self, feature_names):
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
        X_input = pd.DataFrame([weather_features])

        probability = self.model.predict_proba(X_input)[0][1]
        importances = self.model.feature_importances_
        feature_importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        return probability, feature_importances
    
    def get_valid_location(self, location_validator, user_input):
            if (location_validator == 'specific'):
                return extract_location(user_input)
            
            message_map = {
                'different-specific': 'Sorry, it seems you have entered multiple locations, please enter a single location: (e.g. enter a city name or zip code)',
                'unspecific': 'Sorry, I\'m having trouble determinig the location you are inquiring about. Can you please clarify? (e.g. enter a city name or zip code)',
                'none': 'Please specify the location you are inquiring about: (e.g. enter a city name or zip code)'
            }
            new_message = message_map[location_validator]
            new_input = input(new_message).strip()
            new_location_validator = validate_location_info(new_input)
            return self.get_valid_location(new_location_validator, new_input)
    
    def assess_school_policy(self, weather, snow_inches, estimate):
        policy_delta = weather.snow_3h / snow_inches
            # Increase if weather exceeds policy AND estimate is not very high
        if (policy_delta > 1.2 and estimate < 0.8):
            return 0.2
            # Decrease if weather is lower than policy AND if estimate is high
        elif (policy_delta < 0.8 and estimate > 0.7):
            return -0.2
            # Increase if weather is in agreement with policy AND if estimate is not high
        elif (policy_delta >= 0.8 and policy_delta <= 1.2 and estimate < 0.7):
            return 0.2
        
        return 0

    def assess_prior_closure(self, closure_days, probability):
        # according to the data, max closures days per year is 3, with majority at 2 days
        # reduce probability for each day above 2 days
        max_reduction = probability / 2
        if (closure_days == 2):
            return -max_reduction * 0.333
        elif (closure_days == 3):
            return -max_reduction * 0.666
        elif (closure_days >= 4):
            return -max_reduction

        return 0
            
    def reweight_estimate(self, closure_probability, additional_input_validator):
        new_probability = closure_probability
        
        for subarr in additional_input_validator:
            if subarr[0] == 'school_policy':
                new_probability += self.assess_school_policy(subarr[1])
            elif subarr[0] == 'prior_closure':
                new_probability += self.assess_prior_closure(subarr[1], closure_probability)

        return new_probability

    def get_updated_response(self, updated_estimate, estimate):
        response = f"Based on the information you have provided "

        if (updated_estimate > estimate):
            response += f"I estimate the probability of school closure has increased from {estimate} to {updated_estimate}"    
        elif (updated_estimate < estimate):
            response += f"I estimate the probability of school closure has decreased from {estimate} to {updated_estimate}"
        else:
            response += f"I still estimate the probability of school closure is {estimate}"

        return response

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
                
                if location in zip_to_model:
                    model_path = f"./models/{zip_to_model[location]}.pkl"
                else:
                    print(f"I'm sorry - I don't yet have weather data for {location} yet!")
                    break

                self.model = joblib.load(model_path)

                weather_data = self.fetch_weather_data(location)
                closure_probability, factors = self.predict_closure(weather_data)
                response = explain_school_closure(location, weather_data, closure_probability, factors.head(5))

                print(response)

                additional_input = input(
                    f"""
                        I can also udpate my assessment based on additional factors.
                        For example, you can provide the school\'s closure policy or tell me how many days scools has closed for weather thus far this year.

                        Do you want to add anything?
                    """
                ).strip()
                
                additional_input_validator = classify_additional_factors(additional_input)
                updated_estimate = self.reweight_estimate(closure_probability, additional_input_validator)
                updated_response = self.get_updated_response(updated_estimate, closure_probability)
                
                print(updated_response)
                
            user_input = input("Can I help you with anything else?").strip()

if __name__ == "__main__":
    feature_names = [
        "temp", "visibility", "dew_point", "feels_like", "temp_min", "temp_max",
        "pressure", "humidity", "wind_speed", "wind_gust", "wind_deg",
        "rain_1h", "rain_3h", "snow_1h", "snow_3h", "clouds_all", "weather_id",
        "is_weekend", "is_holiday", "is_covid_period"
    ]

    agent = SchoolClosureAgent(feature_names=feature_names)
    agent.handle_interaction()
