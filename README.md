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



User Paths
INITIAL INTERACTION: "How can I help you?" Use LLM to determine if I can help. If I can help and they have included location, go directly to model. Need prompt for location or to let user know I can't help.

Need to figure out best way to control user interactions. Should I have switch statements in my code that sit on top of different prompts? Other ways? 
1) Determine if I can help user with their request - need to prompt telling ai what I can do and asking if i should let user continue with prompt
2) If I can move forward, ask prompt to return location info. Need to check against my data to determine if I have model for that location.
3) Can then run model and return response. Ask if there is any additional info that can help me
4) Is there a way for me to train a model or use a prompt to get a better response from the AI based on additional info?
5) Think about how I can make responses really helpful to user 


Can I do RAG (retreival augmented generation)?