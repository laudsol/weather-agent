DESCRIPTION: 
This agent is designed to help users understand the likelyhood of weather-based school closures in a given district. The agent first provides a statistical estimate of the probability of school closures based on historical weather patterns and previous school closures. Users can then add additional information for the program to consider, such as school closures policy or if school has already closed multiple times that year. This information is evaluated and may change the program's estimate of school closure if it offers a strong confirmation or contradiction of prior assumptions.

MOTIVATION:
I chose this project to get my feet wet with data models and agentic development. More than the coding, this project was a an exercise in thinking through what an agentic applicaiton should be. My early view is that a good project should create a chatGPT-like experience built on real data and properly trained models. Users should be able to query and provide information in human-friendly ways, be led gently down the user-pathways, and receive quantitative but human-freindlly responses. The ultimate goal is to create an experience for users to use quantitative information in the same way most thoughtful statisticians and business analysts do: They don't take an output, like 71% at face value. They want to understand the factors driving that estimate, analyze which are the most influential, and think through various real-world scenarios which could materially change the estimate. An app to answer the question "will school be closed tomorrow?" might a bit contrived for such an example, but I felt it was something simple that real people might use, and, most importantly, doable from a coding perspective.

USAGE:
1) Fork and clone the repo.
2) Obtain an openai API key. Store it in a .env file like so: OPENAI_KEY = "random_key_characters"
3) Download the necessary python packages and run the predict.py file

USER PATHWAYS:
See the diagrams in assets/User_Flow.

I felt it was important to define what the user should and shouldn't be able to do. The idea was to model a traditional UI with set user pathways, while making the user feel they are interacting with an open-ended agent which can take unstructured data inputs.

I have not rigerously implemented every pathway I planned out. I was trying to get a generally working app, not invest in making it production-ready. I did feel it was important to demonstrate that I could implement a recursive user interactions, so the location validator flow works exactly as shown on the diagram.


CONSIDERATIONS:

Data: I only used a single data set for this project. 

Model: I used a RandomForestClassifier to build a model. Frankly, a limear regression would probably have been fine. I was curious to learn about decision trees, and this type of model will certainly be relevant for agentic development. One thing I overlooked at the outset was the small number of positive data points - I used weather data for ~4000 days against only 25 days of weather-related school closure. This error became apparent when I tested the model: no matter how severe the weather input, it predicted 0% probability of closure. When I implemented SMOTE (synthetic minority over-sampling technique) I began to see more intuitive results. In a production app, the model would have to be exmained more closely and tested very robustly.



2) How to prompt users for info
3) Creating defined paths while making it feel like a chat
4) Come up with input classification so I can control the logic
5) Make the user-flow recursive in order to get necessary information
6) Set up different types of catches - failed calls, incorrect data-types, unexpected openAI results, unexpected user input
7) Explain things to the user in a way that makes them feel like there is both quantitative depth and qualitative breadth
8) Productionizing the data would require.... 








