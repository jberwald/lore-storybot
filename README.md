# lore-storybot



## Installation

Set up a Python virtual environment with a newish version of Python, 
version 3.12 or greater should do it. Then in a terminal run

`pip install -r requirements.txt`

This will install all the necessary packages to run the core code found in 
the `storybot` directory, as well as Jupyter-Lab for running the notebook.

Data for the case studies can be found at this 
[repository](https://github.com/nsheils/MLE-case-studies/tree/main), though the
data is already downloaded to the `data` folder in this repo.

## Running the notebook

In a terminal make sure you're working within the venv by running 
`source venv/bin/activate` or something similar if you're using Conda. Then
run `jupyter-lab` to fire up the notebook server. Once running in
browser, open the `bot_runner.ipynb` file. The tutorial is designed to be run through the notebook, so please refer
to it in the Steps below.

## Conversational Evaluation API

I chose to work on Problem Statement 1: 

```text
Problem Statement 1: Conversational Evaluation API

Scenario: Imagine you work for a company that leverages conversational data to
understand how individuals perceive themselves and how these perceptions evolve over
time. These conversations come from one-on-one conversations between a user and a
generative agent (StoryBot). The evolving identity signal drives personalization and
recommendation systems.

Task: Develop a REST API endpoint in Python to evaluate the content of conversations and
determine what the user believes about themselves or a topic. The endpoint should accept
raw conversation data as JSON objects (includes the message and metadata). Consider
feature extraction and apply one or more machine learning models. You may use pre-
trained models (e.g., from Hugging Face) or build a lightweight model yourself. The output
should be in a clear format that can be consumed by different teams to understand the
user’s evolving beliefs about themselves or different topics (e.g., a team assigning value to
conversations, a team developing StoryBot, or a team recommending content in the
community). Provide a brief README outlining how to set up and run your API, along with
instructions for testing the endpoint.
```

### Step 1 - start the API server

The home directory, for the purposes of this tutorial, is the top level of this `lore-storybot` repo. To begin, 
`cd storybot` and run `uvicorn main:app --reload --port 8000`. This will start the primary API server. It utilizes some 
tools and models from Huggingface which may require a few minutes to download.  

### Step 2 - load the data 

From the `bot_runner` notebook, run the first couple of cells to import the required modules and load the data. This application
only used the `conversations` data set. 

### Step 3 - feature extraction and database upload

To mimic the concept that the messages are streaming in, I loop through the `conversations` list. This stage has two
key components. 
1. Sentiment is extracted from each message and attached to the subsequent output.
2. This new 'sentiment-laden' message is uploaded to the database.

Sentiment analysis is achieved in two ways. First, I utilized  the `emotion-english-distilroberta-base` model from 
Huggingface to classify each message's sentiment into one of seven basic sentiment: *anger, disgust, fear,
joy, neutral, sadness, surprise*. The top sentiment is attached to the message going forward. Eg., 
```json
"top_sentiment": {
  "label": "sadness",
  "score": 0.4427511990070343
}
```
Since LLM's are probabilistic models, a second method of sentiment analysis seemed warranted. I chose Python's NLTK library
to arrive at a separate score:
```json
"vader_scores": {
      "neg": 0.065,
      "neu": 0.819,
      "pos": 0.116,
      "compound": 0.296
}
```
These scores come from a rule-based model known as VADER ("Valence Aware Dictionary and sEntiment Reasoner").

Given a list of labels that we are interested in flagging in message, the app aligns the message text with 
those labels using a named entity recognition model (NuNerZero). The current list of labels can be found 
[here](https://github.com/jberwald/lore-storybot/blob/main/storybot/features.py#L135-L138) Here is an example of the `label -> entity` 
mapping:
```json
"entities": {
  "emotion": "down",
  "medical": "health",
  "family": "family"
}
```
(Note: NuNerZero is not available through Huggingface pipelines, so the model must be downloaded by hand to run
the code. In the `storybot` directory, create `numind/NuNerZero` and place the `model.safetensors` and 
`gliner_config.json` in there. Both can be downloaded from [here](https://huggingface.co/numind/NuNER_Zero/tree/main).)

Lastly, the feature extraction step attempted to summarize the messages. This step was fairly ineffective, so was commented out
in the final version. The complete response from the API looks like this:
```json
{
  "response": {
    "vader_scores": {
      "neg": 0.12,
      "neu": 0.813,
      "pos": 0.068,
      "compound": -0.3274
    },
    "top_sentiment": {
      "label": "sadness",
      "score": 0.594653844833374
    },
    "summary": {
      "summary_text": "I managed to get to the stories! But I\u2019m feeling a bit down today. My health has been off, and I\u2019m worried about my family."
    },
    "entities": {
      "emotion": "down",
      "medical": "health",
      "family": "family"
    },
    "metadata": {
      "user_id": 782,
      "conversation_id": 98696,
      "screen_name": "ChattyPenguin",
      "timestamp": "2023-10-02T11:15:00Z"
    }
  }
}
```

After a message is processed for various features, the above response object is uploaded to the database. It should be
noted that the database entry is flattened to work more smoothly with a tabular database. 

### Step 4 - feature analysis

Since end user -- different teams at Lore -- will want access to this data raw or processed data, 
the API also includes a feature analysis component. The goal of this stage is to provide an internal user with insight into the key features of a user's
conversation. In this implementation, a proof of concept TF-IDF model provides the top tokens from a user's conversation.

The database can be searched by `user_id`, which returns all message under that id. If a user has multiple 
conversations, the current implementation will aggregate this token analysis over all of the conversations. 

There are two endpoints associated with this analysis: `/messages` and `/top-features`, which both take `user_id` 
as their only parameter in the payload. The latter calls the first, then performs a straightforward TD-IDF analysis of
user's messages, surfacing the most important words (statistically). Here is an example response:
```json
{'response': 
    {
      'able defend': 4.860729711040595,
      'able help': 4.860729711040595,
      'able home': 4.860729711040595,
      'able make': 4.860729711040595,
      'able offer': 4.860729711040595
    }
}
```
Note that the analysis uses unigrams and bigrams. 

### Looking forward to discussing the project soon.



