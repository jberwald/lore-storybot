from typing import Dict, List

from pydantic import BaseModel
import pandas as pd

class ConversationInput(BaseModel):
    message: str
    metadata: Dict = {}


class Response(BaseModel):
    response: str | dict | List[dict]

class MessageResponse(BaseModel):
    response: List[dict]

class VaderScores(BaseModel):
    neg: float
    neu: float
    pos: float
    compound: float

class TopSentiment(BaseModel):
    label: str
    score: float

class Summary(BaseModel):
    summary_text: str

class Metadata(BaseModel):
    user_id: int
    conversation_id: int
    timestamp: str

class ResponseObject(BaseModel):
    vader_scores: VaderScores
    top_sentiment: TopSentiment
    summary: Summary
    metadata: Metadata

class Entry(BaseModel):
    response: ResponseObject


class UserId(BaseModel):
    user_id: int


def flatten_entry(entry: Entry | dict):
    """
    Turn a nested model into a flattened one.

    FIXME: remove hard-coded keys!
    """
    try:
        entry = entry.model_dump()['response']
    except AttributeError:
        entry = entry['response']
    df = pd.json_normalize(entry, sep="_")
    model = df.to_dict(orient='records')[0]
    # remove the metadata term from normalization
    model['user_id'] = model['metadata_user_id']
    model['conversation_id'] = model['metadata_conversation_id']
    model['timestamp'] = model['metadata_timestamp']
    model['summary_text'] = model['summary_summary_text']
    model.pop('metadata_user_id')
    model.pop('metadata_conversation_id')
    model.pop('metadata_timestamp')
    model.pop('summary_summary_text')
    return model

