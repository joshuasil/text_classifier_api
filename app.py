import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
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

    class_probabilities = loaded_model.predict_proba(new_text)
    top_class_indices = class_probabilities.argsort()[0][::-1][:4]
    top_class_labels = [class_labels[i] for i in top_class_indices]
    return {'message': data,'prediction': top_class_labels}

tepeyac_model = joblib.load('tepeyac_svc_model.pkl')
tepeyac_tfidf_vectorizer = joblib.load('tepeyac_tfidf_vectorizer.pkl')
tepeyac_labels = tepeyac_model.classes_

@app.post('/tepeyacprediction')
def tepeyacprediction(data: Details):
    new_text = [data.text_to_classify]
    print(new_text)
    class_probabilities = tepeyac_model.predict_proba(new_text)
    top_class_indices = class_probabilities.argsort()[0][::-1][:4]
    top_class_labels = [tepeyac_labels[i] for i in top_class_indices]
    return {'message': data,'prediction': top_class_labels}

diabetes_distress_model = joblib.load('diabetes_distress_svc_model.pkl')
diabetes_distress_tfidf_vectorizer = joblib.load('diabetes_distress_tfidf_vectorizer.pkl')
diabetes_distress_labels = diabetes_distress_model.classes_

@app.post('/diabetes_distress_prediction')
def diabetes_distress_prediction(data: Details):
    new_text = [data.text_to_classify]
    print(new_text)
    class_probabilities = diabetes_distress_model.predict_proba(new_text)
    top_class_indices = class_probabilities.argsort()[0][::-1][:4]
    top_class_labels = [diabetes_distress_labels[i] for i in top_class_indices]
    return {'message': data,'prediction': top_class_labels}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=1000, reload=True)