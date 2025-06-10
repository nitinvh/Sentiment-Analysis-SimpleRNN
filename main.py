from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

model = load_model('simple_rnn_imdb.h5')

# mapping word index back to words
word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

# decode function
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

# Preprocess user input

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in text]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)

    return  padded_review

st.title('IMDB movie Review sentiment analysis')
st.write('Enter a movie review to classify it as positive or negative')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)

    output = model.predict(preprocessed_input)

    sentiment ='Positive' if output[0][0]>0.5 else 'Negative'

    st.write('Sentiment: ', sentiment)
    st.write('Score: ', f'{output[0][0]}')

else:
    st.write('Please write a rieview :)')





