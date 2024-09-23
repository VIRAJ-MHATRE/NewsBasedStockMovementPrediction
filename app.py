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
import pickle

NLTK_DATA.data.path.append('./nltk_data')
# Add custom nltk data path if needed
nltk.data.path.append('./nltk_data')  # Adjust the path if necessary

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

import nltk
nltk.download('punkt')

# Load the NER pipeline
ner_pipeline = pipeline('ner', grouped_entities=True)
nltk.data.path.append('./nltk_data')


# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)  # This is the correct resource
nltk.download('averaged_perceptron_tagger_eng', quiet=True)  # This is the correct resource
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)  # VADER lexicon for sentiment analysis

from transformers import pipeline

model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
ner_pipeline = pipeline("ner", model=model_name)
# Load models
@st.cache_resource
def load_xgb_close_class():
    with open('XGB_close_classifier.pkl', 'rb') as file:
        return pickle.load(file)

def load_xgb_close_regg():
    with open('xgb_regressor_close.pkl', 'rb') as file:
        return pickle.load(file)

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

col1, col2 = st.columns(2)
# Input form
# Input form

# Inject CSS to style the text input fields
st.markdown(
    """
    <style>
    .stTextInput > div > input {
        width: 500px;  /* Adjust width as needed */
        ;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .stForm {
        width: 800px;  /* Change to your desired width */
        margin: 0 auto;  /* Center the form */
        height: 800px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    #news-based-stock-movement-predictor {
        width: 1000px; /* Adjust this value as needed */
        margin: 0 auto; /* Center the title */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI setup
st.title('News-Based Stock Movement Predictor.')

st.markdown(
    f"""
    <style>
    .stForm {{
        width: 800; /* Use user-defined width */
        margin-left: 0; /* Align to the left */
        margin-right: 0; /* Remove any right margin */
        height: 100; /* Adjust height as needed */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Create a form
with st.form(key='input_form'):
    col1, col2 = st.columns([3, 1])  # Adjust column widths if needed

    with col1:
        str_data = st.text_area('Please enter the News Article:',height=150)  # Use text_area for better visibility

    with col2:
        Stock_ticker = st.selectbox(
            'Ticker',
            ['-Select-', 'AAPL(Apple)', 'MSFT(Microsoft)', 'NVDA(Nvidia)', 'TSLA(Tesla)', 'AMZN(Amazon)'],
            index=0
        )

    submit_button = st.form_submit_button(label='Predict')


    if submit_button:
        if Stock_ticker == '-Select-':
            st.error('Please select a valid ticker.')
        elif not str_data:
            st.error('Please enter a news article.')
        else:
            # Process the news article
            str_data = str_data.lower()
            df = pd.DataFrame([str_data], columns=['Text'])

            # Assign ticker values
            ticker_mapping = {
                'AAPL(Apple)': 0,
                'MSFT(Microsoft)': 1,
                'NVDA(Nvidia)': 2,
                'TSLA(Tesla)': 3,
                'AMZN(Amazon)': 4
            }

            # Get the ticker value
            ticker_value = ticker_mapping.get(Stock_ticker)

            # Check if the ticker was found
            if ticker_value is None:
                st.error('Invalid ticker selected. Please select a valid ticker.')
                st.stop()

            # Initialize inputdf with the same index as df
            inputdf = pd.DataFrame(index=df.index, columns=[
                'Vader_sentiment_score', 'Blob_polarity', 'BlobSubjectivity',
                'positive_word_count', 'negative_word_count', 'person_count',
                'organization_count', 'location_count', 'anger', 'anticipation',
                'disgust', 'fear', 'joy', 'sadness', 'trust', 'Ticker'
            ])

            # Assign ticker value
            inputdf['Ticker'] = ticker_value

            # Ensure Ticker is numeric
            inputdf['Ticker'] = inputdf['Ticker'].astype(int)

            # Proceed with the rest of your code...

            # Remove stopwords and tokenize
            lemmatizer = WordNetLemmatizer()
            tokenizer = TreebankWordTokenizer()
            stop_words = set(stopwords.words('english'))

            # Apply lemmatization and remove stop words
            df['Text'] = df['Text'].apply(
                lambda x: " ".join([lemmatizer.lemmatize(token, get_pos_tag(tag))
                                    for token, tag in nltk.pos_tag(tokenizer.tokenize(x))
                                    if token not in stop_words]) if pd.notnull(x) else ""
            )

            # VADER sentiment scores
            analyzer = SentimentIntensityAnalyzer()

            def extract_vader_features(text):
                score = analyzer.polarity_scores(text)
                return score['compound']

            inputdf['Vader_sentiment_score'] = df['Text'].apply(extract_vader_features)

            # TextBlob subjectivity and polarity
            def extract_textblob_subjectivity(text):
                blob = TextBlob(text)
                return blob.sentiment.polarity, blob.sentiment.subjectivity

            inputdf['Blob_polarity'], inputdf['BlobSubjectivity'] = zip(*df['Text'].apply(extract_textblob_subjectivity))

            # Positive and negative word counts
            def posneg_vader(text):
                tokens = text.split()
                positive_count = sum(1 for word in tokens if analyzer.polarity_scores(word)['compound'] > 0.05)
                negative_count = sum(1 for word in tokens if analyzer.polarity_scores(word)['compound'] < -0.05)
                return positive_count, negative_count

            inputdf['positive_word_count'], inputdf['negative_word_count'] = zip(*df['Text'].apply(posneg_vader))

            # Name Entity Recognition
            def ner__(text):
                entities = ner_pipeline(text)
                
                # Initialize counts
                person_count = 0
                organization_count = 0
                location_count = 0
                
                # Check what the entities look like
                for entity in entities:
                    print(entity)  # Debug: see what the entity structure is
                    
                    # Adjust based on the actual structure of the entity dictionary
                    if 'entity_group' in entity:
                        if entity['entity_group'] == 'PER':
                            person_count += 1
                        elif entity['entity_group'] == 'ORG':
                            organization_count += 1
                        elif entity['entity_group'] == 'LOC':
                            location_count += 1

                return person_count, organization_count, location_count

            inputdf['person_count'], inputdf['organization_count'], inputdf['location_count'] = zip(*df['Text'].apply(ner__))

            # Emotion scores
            def nrc(text):
                emotion = NRCLex(text)
                scores = emotion.raw_emotion_scores
                return (scores.get('anger', 0), scores.get('anticipation', 0),
                        scores.get('disgust', 0), scores.get('fear', 0),
                        scores.get('joy', 0), scores.get('sadness', 0),
                        scores.get('trust', 0))

            inputdf['anger'], inputdf['anticipation'], inputdf['disgust'], inputdf['fear'], inputdf['joy'], inputdf['sadness'], inputdf['trust'] = zip(*df['Text'].apply(nrc))
            # Make predictions
            
            close_prediction_proba = load_xgb_close_class().predict_proba(inputdf)
            trade_prediction_proba = load_xgb_trade_class().predict_proba(inputdf)
            close_prediction = close_prediction_proba[0, 1]
            negative_close_prediction = close_prediction_proba[0, 0]

            trade_prediction = trade_prediction_proba[0, 1]
            negative_trade_prediction = trade_prediction_proba[0, 0]

            close_regression_prediction = load_xgb_close_regg().predict(inputdf)[0]

            # Display predictions and gauges
    col1, spacer_col, col2 = st.columns([5, 1, 5])  # Adjust the width ratio as needed
        
    # Close Price Gauge
    with col1:
        fig1 = go.Figure(go.Indicator(
            mode="gauge+delta",
            value=(close_prediction * 100 + 0) if close_prediction * 100 > 50 else (0 - close_prediction * 100),
            gauge={
                'axis': {'range': [-100, 100], 'tickvals': []},
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
                }
            },
            delta={'reference': 0, 'position': 'top', 'suffix': ' %'}
        ))

        # Add annotations for 'Bullish' and 'Bearish' at the ends of the gauge
        fig1.add_annotation(
            x=1,  # Right end of the gauge
            y=0,  # Centered vertically
            text="Bullish",
            showarrow=False,
            font=dict(color="green", size=14),
            xref="paper",
            yref="paper",
            xanchor='right',  # Align text to the right
            yanchor='middle'
        )

        fig1.add_annotation(
            x=0,  # Left end of the gauge
            y=0,  # Centered vertically
            text="Bearish",
            showarrow=False,
            font=dict(color="red", size=14),
            xref="paper",
            yref="paper",
            xanchor='left',  # Align text to the left
            yanchor='middle'
        )
        fig1.update_layout(
            title={'text': "Closing Price", 'font': {'size': 24},'xanchor': 'center'},
            title_x=0.5  # Center the title horizontally
        )
    
        # Plot the figure
        st.plotly_chart(fig1)

    # Trade Gauge
    with col2:
        fig2 = go.Figure(
            go.Indicator(
            mode="gauge+delta",
            value=(trade_prediction * 100 + 0) if trade_prediction * 100 > 50 else (0 - trade_prediction * 100),
            gauge={
                'axis': {'range': [-100, 100], 'tickvals': []},
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
                }
            },
            delta={'reference': 0, 'position': 'top', 'suffix': ' %'}
        ))

        # Add text annotation for extremes
        fig2.add_annotation(
            x=1,  # Right end of the gauge
            y=0,  # Centered vertically
            text="Increased",
            showarrow=False,
            font=dict(color="green", size=14),
            xref="paper",
            yref="paper",
            xanchor='right',  # Align text to the right
            yanchor='middle'
        )
        fig2.add_annotation(
            x=0,  # Left end of the gauge
            y=0,  # Centered vertically
            text="Decreased",
            showarrow=False,
            font=dict(color="red", size=14),
            xref="paper",
            yref="paper",
            xanchor='left',  # Align text to the left
            yanchor='middle'
        )
        fig2.update_layout(
            title={'text': "Trade Volume", 'font': {'size': 24},'xanchor': 'center'},
            title_x=0.5  # Center the title horizontally
        )
                # Plot the figure
        st.plotly_chart(fig2)

with st.sidebar:
     
    st.sidebar.markdown("<h1 style='text-align: center;'>Stock Movement Probabilities</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#F65164;" /> """, unsafe_allow_html=True)
    # Close Price Prediction
    coln1, coln2 = st.columns([2, 1])  # Adjust column widths

    if close_prediction *100 > 50:
        with coln1:
            st.markdown("<br><strong>Close Price</strong><br><span style='color: green;'>Bullish chances 📈:</span>", unsafe_allow_html=True)
        with coln2:
            st.metric(label="", value=f"{close_prediction:.2%}", delta="")
            

    else:
        with coln1:
            st.markdown("<br><strong>Close Price</strong><br><span style='color: red;'>Bearish chances 📉:</span>", unsafe_allow_html=True)
        with coln2:
            st.metric(label="", value=f"{negative_close_prediction:.2%}", delta="")
    
    # Trade Volume Prediction
    coln1, coln2 = st.columns([2, 1])  # Adjust column widths

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

    # Close Price Change
    # coln1, coln2 = st.columns([2, 1])  # Adjust column widths

    # if close_regression_prediction > 0:
    #     with coln1:
    #         st.markdown("<br><strong>Close price change</strong><br><span style='color: green;'>Increased 📈</span>", unsafe_allow_html=True)
    #     with coln2:
    #         st.metric(label="", value=f"{close_regression_prediction:.2}%", delta="")
    # else:
    #     with coln1:
    #         st.markdown("<br><strong>Close price change</strong><br><span style='color: red;'>Decreased 📉</span>", unsafe_allow_html=True)
    #     with coln2:
    #         st.metric(label="", value=f"{close_regression_prediction:.2}%", delta="")
