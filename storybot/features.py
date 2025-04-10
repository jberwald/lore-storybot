from dataclasses import dataclass, field
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
from transformers import Pipeline
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except Exception as e:
    nltk.download('vader_lexicon')

from models import ConversationInput

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


@dataclass
class Features:

    classifier: Pipeline = None
    summarizer: Pipeline = None

    def extract_features(self, conversation_data: ConversationInput) -> dict | str:
        """
        Classify the sentiment to one of the following:
            anger
            disgust
            fear
            joy
            neutral
            sadness
            surprise
        """
        try:
            values = self.nltk_sentiment(conversation_data)
            sentiment = self.llm_sentiment(conversation_data)
            summary = self.llm_summarization(conversation_data)
            output = {
                'vader_scores': values,
                'top_sentiment': sentiment,
                'summary': summary
            }
        except Exception as e:
            return f"Error during LLM processing: {e}"
        return output

    def nltk_sentiment(self, conversation_data: ConversationInput) -> dict | str:
        try:
            sentiment = analyzer.polarity_scores(conversation_data.message)
            return sentiment
        except Exception as e:
            return f"Error during LLM processing: {e}"

    def llm_summarization(self, conversation_data: ConversationInput) -> dict | str:
        try:
            output = self.summarizer(conversation_data.message, max_new_tokens=100)
        except Exception as e:
            return f"Error during LLM processing: {e}"

        output = output.pop()
        print("** OUTPUT", output)
        return dict(output)

    def llm_sentiment(self, conversation_data: ConversationInput) -> dict | str:
        try:
            output = self.classifier(conversation_data.message)
        except Exception as e:
            return f"Error during LLM processing: {e}"

        output = output.pop()
        return dict(output)


@dataclass
class MessageProcessor:
    conversation: List[dict] | dict
    messages: List = field(default_factory=list)

    tfidf_matrix: sp.spmatrix = None
    tfidf_vectorizer: TfidfVectorizer = None

    def __post_init__(self):
        # strip the 'response' key off
        if isinstance(self.conversation, dict) and 'response' in self.conversation.keys():
            self.conversation = self.conversation['response']

        self._process_messages()

    def _process_messages(self):
        for msg in self.conversation:
            self.messages.append(msg['message'])

    def fit_tfidf_matrix(self):
        """
        Train a basic TFIDF model to get an idea of token importance in messagesl
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',  # scrub English stop words
            ngram_range=(1, 2), # consider 1 and 2-grams
            token_pattern=r'\b[a-zA-Z]+\b'  # numbers seems to confuse things here
        )
        self.tfidf_vectorizer.fit(self.messages)
        self.tfidf_matrix = self.tfidf_vectorizer.transform(self.messages)

    def get_top_features(self, max_features: int = 5):

        # Tokens, minus those filtered
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        # get the inverse document frequency scores to figure out token importance
        idf_scores = self.tfidf_vectorizer.idf_

        # zip the scores and tokens
        feature_idf_pairs = list(zip(feature_names, idf_scores))

        # Sort the tokens by IDF score
        top_tokens = sorted(feature_idf_pairs, key=lambda item: item[1], reverse=True)
        # Select the top max_featurs tokens, and convert idf values to regular python float
        return {token: float(val) for token, val in top_tokens[:max_features]}








