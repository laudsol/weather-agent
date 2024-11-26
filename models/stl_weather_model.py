import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

weather_data = pd.read_csv('../data/weather_data_stl.csv')
school_closures = pd.read_csv('../data/closure_dates_stl.csv')
holidays = pd.read_csv('../data/holiday_dates.csv')

weather_data['date'] = pd.to_datetime(weather_data['dt'])
school_closures = pd.to_datetime(school_closures['dt'])
holidays = pd.to_datetime(holidays['dt'])

# Create target variable (1 = closed, 0 = open)
weather_data['school_closed'] = weather_data['date'].isin(school_closures).astype(int)

weather_data['is_weekend'] = (weather_data['date'].dt.weekday >= 5).astype(int)
weather_data['is_holiday'] = weather_data['date'].isin(holidays).astype(int)

# Add COVID closure feature
covid_start = pd.Timestamp('2020-03-15')
covid_end = pd.Timestamp('2021-06-30')
weather_data['is_covid_period'] = (
    (weather_data['date'] >= covid_start) & (weather_data['date'] <= covid_end)
).astype(int)

# Assign weights
weather_data['weight'] = 1.0  # Default weight
weather_data.loc[weather_data['is_weekend'] == 1, 'weight'] = 0.1  # Lower weight for weekends
weather_data.loc[weather_data['is_holiday'] == 1, 'weight'] = 0.1  # Lower weight for holidays
weather_data.loc[weather_data['is_covid_period'] == 1, 'weight'] = 0.0  # Ignore COVID period

# convert dt to yyyy-mm-dd format for model
# weather_data['date'] = pd.to_datetime(weather_data['dt'] , unit='s').dt.strftime('%Y-%m-%d')

# Features and target
X = weather_data.drop(['date', 'school_closed', 'weight', 'dt', 'dt_iso' , 'timezone', 'city_name', 'lat', 'lon', 'weather_main', 'weather_description' ,'weather_icon'], axis=1)
y = weather_data['school_closed']
weights = weather_data['weight']

# Split data
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y, weights, test_size=0.2, random_state=42
)

# Train model with weights
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train, sample_weight=weights_train)

# Evaluate model
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

joblib.dump(model, 'school_closure_model.pkl')


model = joblib.load('school_closure_model.pkl')

# Example forecast data
new_data = pd.DataFrame({
    'temp': [27.59],
    'visibility': [0],
    'dew_point': [18.52],
    'feels_like': [18.73],
    'temp_min': [25.88],
    'temp_max': [28.9],
    'pressure': [000],
    'sea_level': [000],
    'grnd_level': [000],
    'humidity': [65],
    'wind_speed': [9.17],
    'wind_deg': [340],
    'wind_gust': [000],
    'rain_1h': [000],
    'rain_3h': [000],
    'snow_1h': [10],
    'snow_3h': [30],
    'clouds_all': [100],
    'weather_id': [602],
    'is_weekend': [0],
    'is_holiday': [0],
    'is_covid_period': [0],
})

# Predict (for scikit-learn)
prediction = model.predict(new_data)
print(f"Prediction (school closed): {prediction}")