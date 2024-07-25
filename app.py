import pandas as pd
import joblib
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import gdown
import os
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")

# Custom CSS to improve aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .main {
        background: #f0f2f6
    }
    .stButton>button {
        color: #ffffff;
        background-color: #4CAF50;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit app
st.title('📊 Stock Price Direction Predictor')

# Function to download and load models
@st.cache_resource
def load_model():
    # URLs for the model and scaler
    model_url = 'https://drive.google.com/uc?id=1c9Ot8xEojZNH9SFKuWil-yr_M_uanHdn'
    scaler_url = 'https://drive.google.com/uc?id=15QFkBkaTBgAMPvaUM3f7GISdw5UtsHCc'
    
    # Download files if they don't exist
    if not os.path.exists('random_forest_model.pkl'):
        with st.spinner('Downloading model... This may take a moment.'):
            gdown.download(model_url, 'random_forest_model.pkl', quiet=False)
    if not os.path.exists('scaler.pkl'):
        with st.spinner('Downloading scaler... This may take a moment.'):
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

# Sidebar for additional information
st.sidebar.header("About")
st.sidebar.info("This app uses a Random Forest model to predict stock price direction. "
                "Enter today's stock data to get a prediction for tomorrow's direction.")
st.sidebar.header("Instructions")
st.sidebar.info("1. Enter the stock data in the input fields.\n"
                "2. Click 'Predict' to see the result.\n"
                "3. The app will show if the stock is likely to go Up (Buy) or Down (Sell).")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Stock Data Input")
    high = st.number_input('High Price', min_value=0.0, format="%.2f")
    low = st.number_input('Low Price', min_value=0.0, format="%.2f")
    open_price = st.number_input('Open Price', min_value=0.0, format="%.2f")
    close = st.number_input('Close Price', min_value=0.0, format="%.2f")
    volume = st.number_input('Volume', min_value=0, step=1)

with col2:
    st.subheader("📊 Data Visualization")
    # Create a simple candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=['Today'],
        open=[open_price],
        high=[high],
        low=[low],
        close=[close]
    )])
    fig.update_layout(title="Today's Stock Data", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

# Create DataFrame from user input
input_data = pd.DataFrame({
    'high': [high],
    'low': [low],
    'open': [open_price],
    'close': [close],
    'volume': [volume]
})

if st.button('🔮 Predict'):
    with st.spinner('Predicting...'):
        prediction = predict_tomorrow(model, scaler, input_data)
    
    if prediction == 1:
        st.success('🚀 Prediction: Up (Buy)')
    else:
        st.error('📉 Prediction: Down (Sell)')

# Display model features (optional)
with st.expander("Show model features"):
    st.write('Features used by the model:')
    st.write(model.feature_names_in_)

# Footer
st.markdown("---")
st.markdown("Created with 💰 by Brian Bruce Bentil")
