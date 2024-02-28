import streamlit as st
import pickle
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

from keras import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

nltk.download('stopwords')

def preprocessing(text):
    text = text.lower()
    text = word_tokenize(text)

    y = []
    for word in text:
        if word.isalnum():
            if word not in stopwords.words('english'):
                ps = SnowballStemmer('english')
                y.append(ps.stem(word))
    return " ".join(y)

tokenizer = pickle.load(open('tokenizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Sentiment Analyser")

input = st.text_area("Enter The Message:")

if st.button("analyze"):

    processed_input = preprocessing(input)

    tokanized_input = tokenizer.texts_to_sequences([processed_input])

    padded_sent = pad_sequences(tokanized_input, padding='post', maxlen=50)

    prediction = model.predict(padded_sent)

    if prediction[0] < 0.5:
        st.header('Negative')
    else:
        st.header('Positive')