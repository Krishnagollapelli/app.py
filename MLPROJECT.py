import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Fetch stock data
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period='2y')
    return df

# Preprocess data
def preprocess_data(df):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['Close']])
    return df_scaled, scaler

# Prepare dataset
def create_sequences(data, time_steps=50):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

# Build LSTM Model
def build_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(50, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Streamlit UI
st.title("Stock Market Prediction")
company = st.selectbox("Select Company", ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"])

if st.button("Predict"):
    df = fetch_stock_data(company)
    df_scaled, scaler = preprocess_data(df)
    X, y = create_sequences(df_scaled)
    
    model = build_model()
    model.fit(X, y, epochs=10, batch_size=16, verbose=1)
    
    prediction = model.predict(X[-1].reshape(1, 50, 1))
    predicted_price = scaler.inverse_transform(prediction)[0][0]
    
    st.write(f"Predicted Closing Price: ${predicted_price:.2f}")
    
    # Candlestick chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=company
    ))
    st.plotly_chart(fig)
