from fastapi import FastAPI
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

# model_id = "thenlper/gte-large"
model_id_summarizer = "meta-llama/Llama-3.2-3B"
# model_id_summarizer = "meta-llama/Llama-3.2-1B"
# model_id_summarizer = "BartForConditionalGeneration"
# model_id = "tabularisai/multilingual-sentiment-analysis"
# model_id = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
model_id_classifier = "j-hartmann/emotion-english-distilroberta-base"
# model_id_summarizer = "mistralai/Mistral-7B-Instruct-v0.3"
# tokenizer = "google-bert/bert-base-cased"

# Load the pre-trained model and tokenizer.
# This should download the model if needed.
classifier = pipeline(
    "text-classification",
    model=model_id_classifier,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    # tokenizer=tokenizer
)

summarizer = pipeline(
    "summarization", # or feature-extraction
    model=model_id_summarizer,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    # tokenizer=tokenizer
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
