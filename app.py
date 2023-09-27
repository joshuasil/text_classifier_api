import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import requests
import json
import joblib
import re
from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings('ignore')


class Details(BaseModel):
    text_to_classify: str

# Save the trained model to a file
model_filename = 'svc_model.pkl'

# Save the TF-IDF vectorizer to a file
tfidf_vectorizer_filename = 'tfidf_vectorizer.pkl'

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        tokens = word_tokenize(text)  # Tokenize text
        return ' '.join(tokens)
    else:
        return ''
    

loaded_model = joblib.load(model_filename)
loaded_tfidf_vectorizer = joblib.load(tfidf_vectorizer_filename)
class_labels = loaded_model.classes_



app = FastAPI()

@app.post('/c4hprediction')
def c4hprediction(data: Details):
    new_text = [data.text_to_classify]
    print(new_text)
    # new_text = [preprocess_text(new_text)]
    print(new_text)
    class_probabilities = loaded_model.predict_proba(new_text)
    top_class_indices = class_probabilities.argsort()[0][::-1][:4]
    top_class_labels = [class_labels[i] for i in top_class_indices]
    return {'message': data,'prediction': top_class_labels}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=1000, reload=True)