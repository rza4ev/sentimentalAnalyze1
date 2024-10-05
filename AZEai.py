import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Downloading the VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Creating the VADER analysis tool
sia = SentimentIntensityAnalyzer()

# Streamlit user interface setup
st.title('Sentiment Analysis with NLTK VADER')
sentence = st.text_input("Enter a sentence to analyze:")

if sentence:
    # Analyzing the sentiment of the input sentence
    sentiment_scores = sia.polarity_scores(sentence)

    # Displaying the original sentence
    st.write(f"Sentence: {sentence}")
    
    # Displaying emojis based on sentiment
    if sentiment_scores['compound'] > 0.4:
        sentiment = "Positive 😀"
    elif sentiment_scores['compound'] < -0.05:
        sentiment = "Negative 😞"
        recomendation='Gobustan National Park is known for its ancient rock carvings and rich archaeological history. Are you interested here?'
    else:
        sentiment = "Neutral 😐"

    # Creating a sidebar to display sentiment scores
    st.sidebar.header("Sentiment Scores")
    st.sidebar.write(f"**Sentiment Interpretation:** {sentiment}")
    st.sidebar.write(f"**Negative Score:** {sentiment_scores['neg']:.2f}")
    st.sidebar.write(f"**Neutral Score:** {sentiment_scores['neu']:.2f}")
    st.sidebar.write(f"**Positive Score:** {sentiment_scores['pos']:.2f}")
    st.sidebar.write(f"**Compound Score:** {sentiment_scores['compound']:.2f}")

    # Displaying the sentiment score result
    st.write(f"Sentiment: {sentiment}")
    st.write(recomendation)