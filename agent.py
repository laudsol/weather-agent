from dotenv import load_dotenv
import openai
import os
import numpy as np

load_dotenv()
api_key = os.getenv("OPENAI_KEY")
openai.api_key = api_key

class RegressionExplainerAgent:
    def __init__(self, model, weather_api):
        self.model = model  # Your trained linear regression model
        self.weather_api = weather_api  # Weather data integration
        self.user_context = {}

    def fetch_weather(self, location):
        """Fetch weather data for the given location."""
        # Replace with your weather API integration
        return {
            "temperature": 30,
            "snowfall": 8,
            "precipitation": "Snow",
            "wind_speed": 10
        }

    def predict_with_explanation(self, weather_data):
        """Make a prediction and extract feature contributions."""
        # Prepare the input features (example feature order)
        input_features = np.array([
            weather_data["temperature"],
            weather_data["snowfall"],
            weather_data["wind_speed"]
        ]).reshape(1, -1)
        
        # Model prediction
        prediction = self.model.predict(input_features)[0]
        
        # Extract feature contributions
        coefficients = self.model.coef_
        intercept = self.model.intercept_
        contributions = coefficients * input_features[0]

        return {
            "prediction": prediction,
            "contributions": contributions,
            "coefficients": coefficients,
            "intercept": intercept
        }

    def generate_llm_response(self, user_input, weather_data, explanation):
        """Use the LLM to generate a user-friendly explanation."""
        prompt = f"""
        You are an assistant helping parents predict school closures based on weather data.
        Here is the forecast:
        - Temperature: {weather_data['temperature']}°F
        - Snowfall: {weather_data['snowfall']} inches
        - Wind Speed: {weather_data['wind_speed']} mph
        - Precipitation: {weather_data['precipitation']}

        The regression model predicts a {explanation['prediction']:.2f} likelihood of school closure.

        The model's explanation:
        - Snowfall contributed {explanation['contributions'][1]:.2f} to the prediction.
        - Temperature contributed {explanation['contributions'][0]:.2f}.
        - Wind speed contributed {explanation['contributions'][2]:.2f}.
        - The baseline prediction (without features) is {explanation['intercept']:.2f}.

        Explain this to the user in simple terms and provide insights into why this prediction was made.
        Respond to the user's query: "{user_input}"
        """

        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": "You are an assistant helping users with school closure predictions."}, {"role": "user", "content": prompt}], max_tokens=300, temperature=0.7)        
        return response['choices'][0]['message']['content']

    def handle_interaction(self):
        print("Welcome to the School Closure Prediction Agent!")
        print("I can help you predict school closures based on weather forecasts.\n")
        
        while True:
            user_input = input("How can I assist you today? (e.g., 'Will schools close tomorrow in Boston?'): ").strip()
            if "exit" in user_input.lower():
                print("Thank you for using the School Closure Prediction Agent. Goodbye!")
                break
            
            # Step 1: Extract location
            location = self.user_context.get("location") or "Boston"
            weather_data = self.fetch_weather(location)
            
            # Step 2: Predict closure and extract feature contributions
            explanation = self.predict_with_explanation(weather_data)
            
            # Step 3: Use LLM to generate a response
            response = self.generate_llm_response(user_input, weather_data, explanation)
            print(f"\n{response}\n")

            # Store location in context for future use
            self.user_context["location"] = location

if __name__ == "__main__":
    # Example regression model (replace with your trained model)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.coef_ = np.array([-0.01, 0.2, 0.05])  # Example coefficients: temperature, snowfall, wind speed
    model.intercept_ = 0.1  # Example intercept
    
    # Initialize the agent with the model and placeholder weather API
    agent = RegressionExplainerAgent(model=model, weather_api=None)
    agent.handle_interaction()
