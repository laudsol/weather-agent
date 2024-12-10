import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
from imblearn.over_sampling import SMOTE

weather_data = pd.read_csv('../data/weather_data_stl.csv')
school_closures = pd.read_csv('../data/closure_dates_stl.csv')
holidays = pd.read_csv('../data/holiday_dates.csv')

school_closures = pd.to_datetime(school_closures['dt']).astype(int) // 10**9
holidays = pd.to_datetime(holidays['dt']).astype(int) // 10**9

daily_weather = weather_data.groupby(pd.to_datetime(weather_data['dt'], unit='s').dt.date).agg({
    'temp': 'mean',
    'visibility': 'mean',
    'dew_point': 'mean',
    'feels_like': 'mean',
    'temp_min': 'min',
    'temp_max': 'max',
    'pressure': 'mean',
    'humidity': 'mean',
    'wind_speed': 'mean',
    'wind_gust': 'max',
    'wind_deg': 'mean',
    'rain_1h': 'sum',
    'rain_3h': 'sum',
    'snow_1h': 'sum',
    'snow_3h': 'sum',
    'clouds_all': 'mean',
    'weather_id': 'max'
}).reset_index()

daily_weather['dt'] = pd.to_datetime(daily_weather['dt']).astype(int) // 10**9

# Create target variable (1 = closed, 0 = open)
daily_weather['school_closed'] = daily_weather['dt'].isin(school_closures).astype(int)
daily_weather['is_weekend'] = pd.to_datetime(daily_weather['dt'], unit='s').dt.weekday.isin([5, 6]).astype(int)
daily_weather['is_holiday'] = daily_weather['dt'].isin(holidays).astype(int)

# Add COVID closure feature
covid_start = int(datetime.strptime('2020-03-15', '%Y-%m-%d').timestamp())
covid_end = int(datetime.strptime('2021-06-30', '%Y-%m-%d').timestamp())
daily_weather['is_covid_period'] = (
    (daily_weather['dt'] >= covid_start) & (daily_weather['dt'] <= covid_end)
).astype(int)

# Fill NaN values with 0 - these are for rain and snow
daily_weather.fillna(0, inplace=True)

# Assign weights
daily_weather['weight'] = 1.0  # Default weight
daily_weather.loc[daily_weather['is_weekend'] == 1, 'weight'] = 0.1  # Lower weight for weekends
daily_weather.loc[daily_weather['is_holiday'] == 1, 'weight'] = 0.1  # Lower weight for holidays
daily_weather.loc[daily_weather['is_covid_period'] == 1, 'weight'] = 0.0  # Ignore COVID period

# Features and target
y = daily_weather['school_closed']
weights = daily_weather['weight']
X = daily_weather.drop(['school_closed', 'dt', 'weight'], axis=1)

# Split data
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y, weights, test_size=0.2, random_state=42
)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train model with balanced data
model = RandomForestClassifier(random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# Evaluate model
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Save
joblib.dump(model, 'school_closure_model.pkl')