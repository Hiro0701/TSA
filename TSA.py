import tensorflow as tf
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

def sentiment_analysis(text):
    model = tf.keras.models.load_model('TSA.h5')
    with open('token.json', 'r') as f:
        tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)

    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=200)

    return round(model.predict(seq)[0])