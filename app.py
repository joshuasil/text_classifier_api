import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import requests
import json



class Details(BaseModel):
    text_to_classify: str

loaded_tokenizer = pickle.load(open('tokenizer.pkl','rb'))

new_model = tf.keras.models.load_model('c4h_text_model')

MAX_SEQUENCE_LENGTH = 125
reverse_mapping_dict = pickle.load(open('reverse_mapping.pkl', 'rb'))


app = FastAPI()

@app.post('/c4hprediction')
def c4hprediction(data: Details):
    new_complaint = [data.text_to_classify]
    
    seq = loaded_tokenizer.texts_to_sequences(new_complaint)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = new_model.predict(padded)[0]
    top_topic_indices = np.argsort(pred)[::-1][:4]
    # print(pred, labels[np.argmax(pred)])
    # Print the top 4 predicted topics and their probabilities
    prediction = [reverse_mapping_dict[idx] for idx in top_topic_indices]

    return {'message': data,'prediction': prediction}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=1000, reload=True)