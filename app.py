import pandas as pd
import joblib
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import gdown
import os

# Streamlit app
st.title('Stock Price Direction Predictor')

# Function to download and load models
@st.cache_resource
def load_model():
    # URLs for the model and scaler
    model_url = 'https://drive.google.com/uc?id=1c9Ot8xEojZNH9SFKuWil-yr_M_uanHdn'
    scaler_url = 'https://drive.google.com/uc?id=15QFkBkaTBgAMPvaUM3f7GISdw5UtsHCc'
    
    # Download files if they don't exist
    if not os.path.exists('random_forest_model.pkl'):
        gdown.download(model_url, 'random_forest_model.pkl', quiet=False)
    if not os.path.exists('scaler.pkl'):
        gdown.download(scaler_url, 'scaler.pkl', quiet=False)
    
    # Load the model and scaler
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# List of predictors used in the model
predictors = ["close", "open", "high", "low", "volume", "ma_5", "ma_10", "ma_20", "volatility", "momentum", "daily_return", "weekly_return"]

# Function to predict stock price direction
def predict_tomorrow(model, scaler, input_data):
    # Add features without rolling calculations
    input_data['ma_5'] = input_data['close']
    input_data['ma_10'] = input_data['close']
    input_data['ma_20'] = input_data['close']
    input_data['volatility'] = 0
    input_data['momentum'] = 0
    input_data['daily_return'] = 0
    input_data['weekly_return'] = 0
    # Ensure the columns are in the correct order
    input_data = input_data[predictors]
    # Scale the features
    input_data_scaled = pd.DataFrame(scaler.transform(input_data), columns=predictors)
    # Predict
    prediction = model.predict(input_data_scaled)
    return prediction[0]

st.write("""
This app predicts the direction of tomorrow's stock price based on today's data.
Please enter the required information below.
""")

# User input
col1, col2 = st.columns(2)
with col1:
    high = st.number_input('High Price', min_value=0.0)
    low = st.number_input('Low Price', min_value=0.0)
    open_price = st.number_input('Open Price', min_value=0.0)
with col2:
    close = st.number_input('Close Price', min_value=0.0)
    volume = st.number_input('Volume', min_value=0, step=1)

# Create DataFrame from user input
input_data = pd.DataFrame({
    'high': [high],
    'low': [low],
    'open': [open_price],
    'close': [close],
    'volume': [volume]
})

if st.button('Predict'):
    prediction = predict_tomorrow(model, scaler, input_data)
    
    if prediction == 1:
        st.success('Prediction: Up (Buy)')
    else:
        st.error('Prediction: Down (Sell)')

# Display model features (optional)
if st.checkbox('Show model features'):
    st.write('Features used by the model:')
    st.write(model.feature_names_in_)
