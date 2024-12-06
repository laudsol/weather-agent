# weather-agent

Tech:
-Model to predict weather 
-Agent to give inputs to model, assess qualitative factors, interact with users

Process:
1) After setting up the data and model, I'm having trouble getting a positive result when running a prediction. I'm testing it with real data from a day school was closed (included in the data set) but it predicts school  will be open. Problem might be data imbalance (only 25 positive cases for more than 4000 data points). 
2) Implemented oversampling using SMOTE. Testing a variety of weather inputs indicated that this approach was certainly directionally correct. Would want more validation in real-life, as well as testing various models for accuracy.



Currently have:
Model will predict probability of closure and return top 5 factors. LLM will format response nicely.

Want:
1) Be able to tell user something like "prbability decreases from 70% to 90% if snow continues like this for 2 more hours" or "probability decreases from 80% to 40% if temperature increases 8 degrees".
2) Asks follow up questions to get better info and update prediction. Like "have there been a lot fo school closures so far this year?" or "can you provide me with the school closure policy?"


Store user inputs for later....
Deal with catches