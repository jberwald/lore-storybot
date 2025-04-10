from fastapi import FastAPI, HTTPException
from tinydb import TinyDB, Query

from models import (
    Entry,
    Response,
    flatten_entry,
    UserId,
)
from features import MessageProcessor


app = FastAPI()

# Database file name
DATABASE_FILE = "./storybot.json"

# Create the file if it doesn't exist)
db = TinyDB(DATABASE_FILE)
table = db.table('conversations')


@app.post("/entry", response_model=Response)
async def create_entry(entry: Entry):
    # get a flattened dictionary
    entry = flatten_entry(entry)
    print("entry", entry)
    try:
        entry_id = table.insert(entry)
        return {"response": {"id": entry_id}}
    except Exception as e:
        return {"response": f"Exception {e}"}


@app.get("/entry/{entry_id}")
async def read_entry(entry_id: int):
    try:
        entry = table.get(doc_id=entry_id)
        if entry:
            return {"id": entry.doc_id, **entry}
        else:
            raise HTTPException(status_code=404, detail="entry not found")
    except Exception as e:
        raise (HTTPException(status_code=500, detail=f"Database error: {e}"))


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
