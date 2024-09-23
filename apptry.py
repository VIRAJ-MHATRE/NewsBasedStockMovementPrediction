import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from transformers import pipeline
import os
import pickle

# Load the NER pipeline
ner_pipeline = pipeline('ner', grouped_entities=True)

# Assuming this code is in the same directory as 'general data models'
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'general data models')

# Load models
@st.cache_resource
def load_xgb_close_class():
    with open(os.path.join(MODEL_DIR, 'XGB_close_classifier.pkl'), 'rb') as file:
        return pickle.load(file)

def load_xgb_close_regg():
    with open(os.path.join(MODEL_DIR, 'xgb_regressor_close.pkl'), 'rb') as file:
        return pickle.load(file)

def load_xgb_trade_class():
    with open(os.path.join(MODEL_DIR, 'XGB_trade_classifier.pkl'), 'rb') as file:
        return pickle.load(file)

# Streamlit UI setup
close_prediction = 0.00
negative_close_prediction = 0.00
trade_prediction = 0.00
negative_trade_prediction = 0.00
close_regression_prediction = 0.00

col1, col2 = st.columns(2)

# Create a form
with st.form(key='input_form'):
    col1, col2 = st.columns([3, 1])

    with col1:
        str_data = st.text_area('Please enter the News Article:')

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
            # Process the news article and make predictions...
            # (Code for processing input and generating predictions goes here)

            # Display predictions and gauges
            col1, col2 = st.columns(2)

            # Close Price Gauge
            with col1:
                import plotly.graph_objects as go

            
                # Value for the gauge
                value = close_prediction - 100  # Map 0-100 to -100 to 0

                fig1 = go.Figure(go.Indicator(
                    mode="gauge+delta",
                    value=value,
                    title={'text': "Closing Price"},
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
                            'value': 65  # Adjust this threshold as needed
                        }
                    },
                    delta={'reference': 0, 'position': 'top', 'suffix': ' %'}
                ))

                # Add annotations for 'Bullish' and 'Bearish'
                fig1.add_annotation(
                    x=1,  # Right end of the gauge
                    y=0,
                    text="Bullish",
                    showarrow=False,
                    font=dict(color="green", size=14),
                    xref="paper",
                    yref="paper",
                    xanchor='right',
                    yanchor='middle'
                )

                fig1.add_annotation(
                    x=0,  # Left end of the gauge
                    y=0,
                    text="Bearish",
                    showarrow=False,
                    font=dict(color="red", size=14),
                    xref="paper",
                    yref="paper",
                    xanchor='left',
                    yanchor='middle'
                )

                # Display the gauge chart
                st.plotly_chart(fig1)


            # Trade Gauge
            with col2:
                fig2 = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=(trade_prediction * 200) - 100,  # Scale from -100 to 100
                    title={'text': "Trade Classifier"},
                    gauge={
                        'axis': {'range': [-100, 100]},
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
                    number={'suffix': '%'},
                    delta={'reference': 0, 'position': 'top'}
                ))

                st.plotly_chart(fig2)

            with st.sidebar:
                st.sidebar.markdown("<h1 style='text-align: center;'>Stock Movement Analysis</h1>", unsafe_allow_html=True)
                st.sidebar.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#F65164;" /> """, unsafe_allow_html=True)

                # Close Price Prediction
                coln1, coln2 = st.columns([2, 1])

                if close_prediction > negative_close_prediction:
                    with coln1:
                        st.markdown("<br><strong>Close Price</strong><br><span style='color: green;'>Bullish chances 📈</span>", unsafe_allow_html=True)
                    with coln2:
                        st.metric(label="", value=f"{close_prediction:.2%}", delta="")
                else:
                    with coln1:
                        st.markdown("<br><strong>Close Price</strong><br><span style='color: red;'>Bearish chances 📉</span>", unsafe_allow_html=True)
                    with coln2:
                        st.metric(label="", value=f"{close_prediction:.2%}", delta="")
                
                # Trade Volume Prediction
                coln1, coln2 = st.columns([2, 1])

                if trade_prediction > negative_trade_prediction:
                    with coln1:
                        st.markdown("<br><strong>Trade</strong><br><span style='color: green;'>Volume increase by:</span>", unsafe_allow_html=True)
                    with coln2:
                        st.metric(label="", value=f"{trade_prediction:.2%}", delta="")
                else:
                    with coln1:
                        st.markdown("<br><strong>Trade</strong><br><span style='color: red;'>Volume decrease by:</span>", unsafe_allow_html=True)
                    with coln2:
                        st.metric(label="", value=f"{trade_prediction:.2%}", delta="")

                # Close Price Change
                coln1, coln2 = st.columns([2, 1])

                if close_regression_prediction > 0:
                    with coln1:
                        st.markdown("<br><strong>Close price change</strong><br><span style='color: green;'>Increased 📈</span>", unsafe_allow_html=True)
                    with coln2:
                        st.metric(label="", value=f"{close_regression_prediction:.2}%", delta="")
                else:
                    with coln1:
                        st.markdown("<br><strong>Close price change</strong><br><span style='color: red;'>Decreased 📉</span>", unsafe_allow_html=True)
                    with coln2:
                        st.metric(label="", value=f"{close_regression_prediction:.2}%", delta="")
