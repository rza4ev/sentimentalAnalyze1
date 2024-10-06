import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure the VADER lexicon is available
nltk.download('vader_lexicon')

# Create the VADER sentiment analysis tool
sia = SentimentIntensityAnalyzer()

# Streamlit user interface setup
st.title('Sentiment Analysis with NLTK VADER')
sentence = st.text_input("Enter a sentence to analyze:")

if sentence:
    # Analyzing the sentiment of the input sentence
    sentiment_scores = sia.polarity_scores(sentence)

    # Determining sentiment and generating appropriate recommendation
    if sentiment_scores['compound'] > 0.4:
        sentiment = "Positive 😀"
        recommendation = "We're happy you're feeling good! Thanks for using our sentiment analyzer."
    elif sentiment_scores['compound'] < -0.05:
        sentiment = "Negative 😞"
        if 'bike' in sentence.lower():  # Check if 'bicycle' is mentioned, case-insensitively
            recommendation = "It seems like you're interested in outdoor activities. Why not explore the camping areas next time?"
        else:
            recommendation = "It seems like you're having a tough time. Did you know about Gobustan National Park and its ancient rock carvings? It could be a refreshing place to visit!"
    else:
        sentiment = "Neutral 😐"
        recommendation = "Thanks for using our sentiment analyzer. We hope you find this tool useful."

    # Displaying the original sentence, sentiment, and recommendation
    st.write(f"Sentence: {sentence}")
    st.write(f"Sentiment: {sentiment}")
    st.write(recommendation)

    # Creating a sidebar to display sentiment scores
    st.sidebar.header("Sentiment Scores")
    st.sidebar.write(f"**Sentiment Interpretation:** {sentiment}")
    st.sidebar.write(f"**Negative Score:** {sentiment_scores['neg']:.2f}")
    st.sidebar.write(f"**Neutral Score:** {sentiment_scores['neu']:.2f}")
    st.sidebar.write(f"**Positive Score:** {sentiment_scores['pos']:.2f}")
    st.sidebar.write(f"**Compound Score:** {sentiment_scores['compound']:.2f}")
