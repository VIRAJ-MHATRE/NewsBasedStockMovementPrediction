import nltk
import pickle
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import TreebankWordTokenizer
from nrclex import NRCLex
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import os
import gc  # Import garbage collection

# Initialize NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Load the NER pipeline
ner_pipeline = pipeline('ner', grouped_entities=True)
nltk.data.path.append('./nltk_data')

# Load models with caching
@st.cache_resource
def load_xgb_close_class():
    with open('XGB_close_classifier.pkl', 'rb') as file:
        return pickle.load(file)

@st.cache_resource
def load_xgb_close_regg():
    with open('xgb_regressor_close.pkl', 'rb') as file:
        return pickle.load(file)

@st.cache_resource
def load_xgb_trade_class():
    with open('XGB_trade_classifier.pkl', 'rb') as file:
        return pickle.load(file)

def get_pos_tag(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

close_prediction = 0.00
negative_close_prediction = 0.00
trade_prediction = 0.00
negative_trade_prediction = 0.00
close_regression_prediction = 0.00

# Streamlit UI setup
st.title('News-Based Stock Movement Predictor.')

# Input form
with st.form(key='input_form'):
    col1, col2 = st.columns([3, 1])
    str_data = st.text_area('Please enter the News Article:', height=150)
    Stock_ticker = st.selectbox('Ticker', ['-Select-', 'AAPL(Apple)', 'MSFT(Microsoft)', 'NVDA(Nvidia)', 'TSLA(Tesla)', 'AMZN(Amazon)'], index=0)
    submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        if Stock_ticker == '-Select-':
            st.error('Please select a valid ticker.')
        elif not str_data:
            st.error('Please enter a news article.')
        else:
            str_data = str_data.lower()
            df = pd.DataFrame([str_data], columns=['Text'])

            ticker_mapping = {'AAPL(Apple)': 0, 'MSFT(Microsoft)': 1, 'NVDA(Nvidia)': 2, 'TSLA(Tesla)': 3, 'AMZN(Amazon)': 4}
            ticker_value = ticker_mapping.get(Stock_ticker)

            if ticker_value is None:
                st.error('Invalid ticker selected. Please select a valid ticker.')
                st.stop()

            inputdf = pd.DataFrame(index=df.index, columns=[
                'Vader_sentiment_score', 'Blob_polarity', 'BlobSubjectivity',
                'positive_word_count', 'negative_word_count', 'person_count',
                'organization_count', 'location_count', 'anger', 'anticipation',
                'disgust', 'fear', 'joy', 'sadness', 'trust', 'Ticker'
            ])
            inputdf['Ticker'] = ticker_value

            # Tokenization and lemmatization
            lemmatizer = WordNetLemmatizer()
            tokenizer = TreebankWordTokenizer()
            stop_words = set(stopwords.words('english'))

            df['Text'] = df['Text'].apply(
                lambda x: " ".join([lemmatizer.lemmatize(token, get_pos_tag(tag))
                                    for token, tag in nltk.pos_tag(tokenizer.tokenize(x))
                                    if token not in stop_words]) if pd.notnull(x) else ""
            )

            # VADER sentiment scores
            analyzer = SentimentIntensityAnalyzer()
            inputdf['Vader_sentiment_score'] = df['Text'].apply(lambda text: analyzer.polarity_scores(text)['compound'])

            # TextBlob sentiment
            inputdf['Blob_polarity'], inputdf['BlobSubjectivity'] = zip(*df['Text'].apply(lambda text: (TextBlob(text).sentiment.polarity, TextBlob(text).sentiment.subjectivity)))

            # Positive and negative word counts
            inputdf['positive_word_count'], inputdf['negative_word_count'] = zip(*df['Text'].apply(lambda text: (
                sum(1 for word in text.split() if analyzer.polarity_scores(word)['compound'] > 0.05),
                sum(1 for word in text.split() if analyzer.polarity_scores(word)['compound'] < -0.05)
            )))

            # NER
            inputdf['person_count'], inputdf['organization_count'], inputdf['location_count'] = zip(*df['Text'].apply(lambda text: (
                sum(1 for entity in ner_pipeline(text) if entity['entity_group'] == 'PER'),
                sum(1 for entity in ner_pipeline(text) if entity['entity_group'] == 'ORG'),
                sum(1 for entity in ner_pipeline(text) if entity['entity_group'] == 'LOC')
            )))

            # Emotion scores
            emotion_scores = df['Text'].apply(lambda text: NRCLex(text).raw_emotion_scores).tolist()

            # Ensure that emotion_scores is a list of dictionaries with the same number of rows as inputdf
            emotion_df = pd.DataFrame(emotion_scores)

            # Fill missing columns with 0 if any emotion is not present
            for emotion in ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'trust']:
                if emotion not in emotion_df:
                    emotion_df[emotion] = 0

            # Assign the columns to inputdf
            inputdf[['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'trust']] = emotion_df[['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'trust']]

            # Display predictions
            col1, col2 = st.columns([5, 5])
            with col1:
                fig1 = go.Figure(go.Indicator(
                    mode="gauge+delta",
                    value=(close_prediction * 100 + 0) if close_prediction * 100 > 50 else (0 - close_prediction * 100),
                    gauge={'axis': {'range': [-100, 100]},
                          'bar': {'color': 'black'},
                          'steps': [
                              {'range': [-100, -20], 'color': 'red'},
                              {'range': [-20, 0], 'color': 'orange'},
                              {'range': [0, 20], 'color': 'yellow'},
                              {'range': [20, 100], 'color': 'lightgreen'},
                          ],
                          'threshold': {
                              'line': {'color': "blue", 'width': 4},
                              'thickness': 1,
                              'value': 65
                          }},
                    delta={'reference': 0, 'position': 'top', 'suffix': ' %'}
                ))
                fig1.update_layout(title={'text': "Closing Price", 'font': {'size': 24}, 'xanchor': 'center'}, title_x=0.5)
                st.plotly_chart(fig1)

            with col2:
                fig2 = go.Figure(go.Indicator(
                    mode="gauge+delta",
                    value=(trade_prediction * 100 + 0) if trade_prediction * 100 > 50 else (0 - trade_prediction * 100),
                    gauge={'axis': {'range': [-100, 100]},
                          'bar': {'color': 'black'},
                          'steps': [
                              {'range': [-100, -20], 'color': 'red'},
                              {'range': [-20, 0], 'color': 'orange'},
                              {'range': [0, 20], 'color': 'yellow'},
                              {'range': [20, 100], 'color': 'lightgreen'},
                          ],
                          'threshold': {
                              'line': {'color': "blue", 'width': 4},
                              'thickness': 1,
                              'value': 65
                          }},
                    delta={'reference': 0, 'position': 'top', 'suffix': ' %'}
                ))
                fig2.update_layout(title={'text': "Trade Volume", 'font': {'size': 24}, 'xanchor': 'center'}, title_x=0.5)
                st.plotly_chart(fig2)

            # Clear memory
            del df, inputdf
            gc.collect()

with st.sidebar:
    st.sidebar.markdown("<h1 style='text-align: center;'>Stock Movement Probabilities</h1>", unsafe_allow_html=True)
    coln1, coln2 = st.columns([2, 1])

    if close_prediction * 100 > 50:
        with coln1:
            st.markdown("<strong>Close Price</strong><br><span style='color: green;'>Bullish chances 📈:</span>", unsafe_allow_html=True)
        with coln2:
            st.metric(label="", value=f"{close_prediction:.2%}", delta="")
    else:
        with coln1:
            st.markdown("<strong>Close Price</strong><br><span style='color: red;'>Bearish chances 📉:</span>", unsafe_allow_html=True)
        with coln2:
            st.metric(label="", value=f"{negative_close_prediction:.2%}", delta="")

    if trade_prediction * 100 > 50:
        with coln1:
            st.markdown("<strong>Trade</strong><br><span style='color: green;'>Volume increase chances📊:</span>", unsafe_allow_html=True)
        with coln2:
            st.metric(label="", value=f"{trade_prediction:.2%}", delta="")
    else:
        with coln1:
            st.markdown("<strong>Trade</strong><br><span style='color: red;'>Volume decrease chances📊:</span>", unsafe_allow_html=True)
        with coln2:
            st.metric(label="", value=f"{negative_trade_prediction:.2%}", delta="")
