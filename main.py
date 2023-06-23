from flask import Flask, jsonify, request
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

model = tf.keras.models.load_model('utils/model/bene.h5')
dataset = pd.read_csv('utils/dataset/tweets_clean.csv')
model.summary()

app = Flask(__name__)


data = dataset.drop(['date'], axis=1)
data = data[data['label'] < 3]
data['label'] = data['label'].replace([0.0, 1.0, 2.0], ['Not Related', 'Kebakaran', 'Pencegahan'])
label = pd.get_dummies(data.label)
data_baru = pd.concat([data, label], axis=1)
data_baru = data_baru.drop(columns='label')
tweet = data_baru['tweet'].values
label = data_baru[['Kebakaran', 'Not Related', 'Pencegahan']].values



X_train, X_test, y_train, y_test = train_test_split(tweet, label, test_size=0.2, random_state=123)

max_word = 6000
tokenizer = Tokenizer(num_words=max_word, oov_token='x')
tokenizer.fit_on_texts(X_train)
tokenizer.fit_on_texts(X_test)

maxlen = 40
teks = "Kebakaran gasi bg"

teks_sequence = tokenizer.texts_to_sequences([teks])
teks_padded = pad_sequences(teks_sequence, maxlen=maxlen, padding='post', truncating='post', value=0)
kategori = model.predict(teks_padded)
label_encoder = np.argmax(kategori, axis=-1)
label_encoder = np.vectorize({0: 'Tidak kebakaran', 1: 'Kebakaran', 2: 'Penanganan'}.get)(label_encoder)
print("Input teks: ", teks)
print("Hasil prediksi: ", label_encoder[0])


@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        tweet = request.form['tweet']
        sequence = tokenizer.texts_to_sequences([tweet])
        padded = pad_sequences(sequence, maxlen=maxlen, padding='post', truncating='post', value=0) 
        category = model.predict(padded)
        predicted = np.argmax(category, axis=-1)
        predicted = np.vectorize({0: 'tidak kebakaran', 1: 'kebakaran', 2: 'penanganan'}.get)(predicted)
        return {"data": {"tweet": tweet,"predict":predicted[0]}}
    except:
        return {"data": {"tweet": tweet,"predict":"error"}}


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
