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

Model: I used a RandomForestClassifier to build a model. Frankly, a linear regression would probably have been fine. I was curious to learn about decision trees, and this type of model will certainly be relevant for agentic development. One thing I overlooked at the outset was the small number of positive data points - I used weather data for ~4000 days against only 25 days of weather-related school closure. This error became apparent when I tested the model: no matter how severe the weather input, it predicted 0% probability of closure. When I implemented SMOTE (synthetic minority over-sampling technique) I began to see more intuitive results. In a production app, the model would have to be exmained more closely and tested very robustly.

Prompting: At the outset I didn't have an intiutive grasp for how to prompt users. I thought I should ask directly for a zip code and use regex to validate user input. This would have been a good idea for a web-based UI, but agentic apps probably need a different paradigm: the whole point is to give the user the sense that they are interacting with an LLM. I therefore built prompts in conjunction with the LLM to define geenral user pathways, but without giving the user the sense that they are being strictly controlled (e.g. I wanted to avoid creating user menus, and having the user select menu items). Once I figured out that I should write LLM queries to define user intent the process became more clear. 

Catches: I've had to consider several types of catches, since user input and returns from API calls are very loose (unlike a traditional API)
1) Failed API calls
2) Unexpected data structures, data types, or values returned from prompts to openai's API.
3) Unexpected user input (ambigious or irrelevant). Not handled as a clear catch - use LLM to clasify intent and turn these into structured inputs.

Testing:
To get the most out of API development, I focused tests on trying various queries to openai. The nature of openai's product means that I cannot expect precisely the same results every time. But I wanted to ensure that the propts are directionally correct. I could foresee long-term difficulties with this approach, and it will be important to keep refining the prompts to narrow the specificity.

PRODUCTION:

Here are a few action-items needed to make the project ready for production.
1) Data pipeline
    - uniform source for weather data
    - automated methods for getting school clsoure data (scraping twitter???)
    - data validation
2) Models
    - Need to consider the various problems which could arrise in training models.
    - Probably need a dynamic way to use various methods in training models, without messing with the code every time changes are needed. For example, could I have added SMOTE wihtout changing the code on the server?
    - Consider writing suite of automated test to check each model as it is trained. Scenarios would need to be thought through since the data changes every time
3) Consider storing openai promts in a database. Might need to mess with these a lot later on. Consider setting up an admin endpoint to update the prompts, as well as UI for the APIs, like Swagger
4) Clean up the user experience based on the user-flow diagaram. Also add pathways to allow user to check what locations are available.