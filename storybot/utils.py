import requests
import threading
import json

from .models import Entry


feature_url = "http://127.0.0.1:8000/extract_sentiment"
entry_url = "http://127.0.0.1:8000/entry"
headers = {
    "Content-Type": "application/json"
}


def hit_api(conversation: list, verbose: bool = False):
    try:
        for message in conversation['messages_list']:
            # Inefficient multiple lookup
            user_id = message['ref_user_id']
            conversation_id = message['ref_conversation_id']
            screen_name = message['screen_name']

            if message['ref_user_id'] == 1:
                continue

            payload = {
                "message": message['message'],
                "metadata": {
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "screen_name": screen_name,
                    "timestamp": message['transaction_datetime_utc']
                }
            }
            try:
                feature_response = requests.post(feature_url, headers=headers, json=payload)

                # Check if the request was successful (status code 2xx)
                if 200 <= feature_response.status_code < 300:
                    if verbose:
                        print("DB API Response:")
                        print(json.dumps(feature_response.json(), indent=2))
                    # Now that we have a feature_response, upload it to the DB
                    try:
                        item = Entry(**feature_response.json())
                        db_response = requests.post(entry_url, headers=headers, data=item.model_dump_json())

                        if 200 <= db_response.status_code < 300:
                            print(f"Upload message with id {id}")
                        else:
                            print(f"Error: DB API request failed with status code {db_response.status_code}")
                            print(f"Response content: {db_response.text}")
                    except requests.exceptions.RequestException as e:
                        print(f"An error occurred during the request: {e}")
                else:
                    print(f"Error: API request failed with status code {feature_response.status_code}")
                    print(f"Response content: {feature_response.text}")

            except requests.exceptions.RequestException as e:
                print(f"An error occurred during the request: {e}")

    except requests.exceptions.RequestException as e:
        print(f"Error hitting API: {e}")


def upload_messages(conversations: list, num_conversations: int = 10):
    """
    Upload and process conversations in parallel
    """
    threads = []
    for conv in conversations[:num_conversations]:
        thread = threading.Thread(target=hit_api, args=(conv,))
        threads.append(thread)
        thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        print(f"Finished uploading {len(conversations[:num_conversations])} conversations!")
