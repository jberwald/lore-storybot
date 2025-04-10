from fastapi import FastAPI, HTTPException
from transformers import pipeline
import torch

if torch.backends.mps.is_available():
    print("MPS is available!")
    mps_device = torch.device("mps")
else:
    print("MPS is not available on this device.")
    mps_device = torch.device("cpu") # Fallback to CPU if MPS is not available

from models import ConversationInput, Response
from features import Features

from tinydb import TinyDB, Query

from models import (
    Entry,
    Response,
    flatten_entry,
    UserId,
)
from features import MessageProcessor


# Various models tested and considered.
# model_id = "thenlper/gte-large"
model_id_summarizer = "meta-llama/Llama-3.2-3B"
# model_id_summarizer = "meta-llama/Llama-3.2-1B"
# model_id_summarizer = "BartForConditionalGeneration"
# model_id = "tabularisai/multilingual-sentiment-analysis"
# model_id = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
model_id_classifier = "j-hartmann/emotion-english-distilroberta-base"
# model_id_summarizer = "mistralai/Mistral-7B-Instruct-v0.3"
# tokenizer = "google-bert/bert-base-cased"

# Set up the Database
# Database file name
DATABASE_FILE = "./storybot.json"

# Create the file if it doesn't exist)
db = TinyDB(DATABASE_FILE)
table = db.table('conversations')


# Load the pre-trained model.
# This should download the model if needed.
classifier = pipeline(
    "text-classification",
    model=model_id_classifier,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

summarizer = pipeline(
    "summarization", # or feature-extraction
    model=model_id_summarizer,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

app = FastAPI()
ftr = Features(
    classifier=classifier,
    summarizer=summarizer
)


@app.post("/extract_sentiment", response_model=Response)
async def extract_sentiment(conversation_data: ConversationInput) -> dict:
    """
    Endpoint to extract sentiment from a request
    """
    try:
        output = ftr.extract_features(conversation_data)
        output['metadata'] = conversation_data.metadata
        return {"response": output}
    except Exception as e:
        return {"response": f"Exception {e}"}


@app.post("/entry", response_model=Response)
async def create_entry(entry: Entry):
    # get a flattened dictionary
    entry = flatten_entry(entry)
    try:
        entry_id = table.insert(entry)
        return {"response": {"id": entry_id}}
    except Exception as e:
        return {"response": f"Exception {e}"}


@app.post("/messages", response_model=Response)
async def messages(user_id: UserId):
    """
    Get all of a user's message.
    """
    try:
        user_messages = search_table(user_id)
        return {"response": user_messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@app.post("/top-features", response_model=Response)
async def top_features(user_id: UserId):
    try:
        msgs = search_table(user_id)

        mp = MessageProcessor(msgs)
        mp.fit_tfidf_matrix()
        tokens = mp.get_top_features()
        return {'response': tokens}
    except Exception as e:
        return {"response": f"Exception {e}"}


def search_table(user_id: UserId) -> list:
    User = Query()
    try:
        entries = table.search(User.user_id == user_id.user_id)
        if entries:
            user_messages = [
                {
                    'conversation_id': entry['conversation_id'],
                    'message': entry['summary_text']
                }
                for entry in entries
            ]
            return user_messages
        else:
            return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
