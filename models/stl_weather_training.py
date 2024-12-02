import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime


weather_data = pd.read_csv('../data/weather_data_stl.csv')
school_closures = pd.read_csv('../data/closure_dates_stl.csv')
holidays = pd.read_csv('../data/holiday_dates.csv')

school_closures = pd.to_datetime(school_closures['dt']).astype(int) // 10**9
holidays = pd.to_datetime(holidays['dt']).astype(int) // 10**9

# daily_weather = weather_data.groupby(weather_data['date'].dt.date).agg({
#     'temp': 'mean',
#     'visibility': 'mean',
#     'dew_point': 'mean',
#     'feels_like':   'mean',
#     'temp_min': 'min',
#     'temp_max': 'max',
#     'pressure': 'mean',
#     'humidity': 'mean',
#     'wind_speed': 'mean',
#     'wind_gust': 'max',
#     'rain_1h': 'sum',
#     'snow_1h': 'sum',
#     'clouds_all': 'mean',
#     'weather_id': 'max'
# }).reset_index()

# Create target variable (1 = closed, 0 = open)
weather_data['school_closed'] = weather_data['dt'].isin(school_closures).astype(int)
weather_data['is_weekend'] = pd.to_datetime(weather_data['dt'], unit='s').dt.weekday.isin([5, 6]).astype(int)
weather_data['is_holiday'] = weather_data['dt'].isin(holidays).astype(int)

# Add COVID closure feature
covid_start = int(datetime.strptime('2020-03-15', '%Y-%m-%d').timestamp())
covid_end = int(datetime.strptime('2021-06-30', '%Y-%m-%d').timestamp())
weather_data['is_covid_period'] = (
    (weather_data['dt'] >= covid_start) & (weather_data['dt'] <= covid_end)
).astype(int)

# Assign weights
weather_data['weight'] = 1.0  # Default weight
weather_data.loc[weather_data['is_weekend'] == 1, 'weight'] = 0.1  # Lower weight for weekends
weather_data.loc[weather_data['is_holiday'] == 1, 'weight'] = 0.1  # Lower weight for holidays
weather_data.loc[weather_data['is_covid_period'] == 1, 'weight'] = 0.0  # Ignore COVID period


# Features and target
X = weather_data.drop(['school_closed', 'weight', 'dt', 'dt_iso' , 'timezone', 'city_name', 'lat', 'lon', 'weather_main', 'weather_description' ,'weather_icon'], axis=1)
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