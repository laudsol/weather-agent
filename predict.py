import pandas as pd
import joblib

model = joblib.load('./models/school_closure_model.pkl')

# Example forecast data
new_data = pd.DataFrame({
    'temp': [25.6],
    'visibility': [402],
    'dew_point': [22.41],
    'feels_like': [13.0],
    'temp_min': [24.4],
    'temp_max': [26.4],
    'pressure': [1020],
    'sea_level': [000],
    'grnd_level': [000],
    'humidity': [86],
    'wind_speed': [18.41],
    'wind_deg': [70],
    'wind_gust': [000],
    'rain_1h': [000],
    'rain_3h': [000],
    'snow_1h': [4.1],
    'snow_3h': [7.1],
    'clouds_all': [100],
    'weather_id': [602],
    'is_weekend': [0],
    'is_holiday': [0],
    'is_covid_period': [0],
})

prediction = model.predict(new_data)
print(f"Prediction (school closed): {prediction}")