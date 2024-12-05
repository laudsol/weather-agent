# weather-agent

Tech:
-Model to predict weather 
-Agent to give inputs to model, assess qualitative factors, interact with users

Process:
1) After setting up the data and model, I'm having trouble getting a positive result when running a prediction. I'm testing it with real data from a day school was closed (included in the data set) but it predicts school  will be open. Problem might be data imbalance (only 25 positive cases for more than 4000 data points). 
2) Implemented oversampling using SMOTE. Initial test predicted school closure using real closure case. Need to test more thouroughly.
