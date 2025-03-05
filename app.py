import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import bcrypt
import mysql.connector
import time
from datetime import datetime

# Set Page Configuration
st.set_page_config(page_title="Stock Market App", layout="wide")

# API Key & Cache
API_KEY = "YOUR_ALPHAVANTAGE_API_KEY"
stock_cache = {}

# Database Connection
def create_connection():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",  # Replace with your MySQL username
            password="your_password",  # Replace with your MySQL password
            database="stock_market_app"
        )
    except mysql.connector.Error as e:
        st.error(f"Error connecting to MySQL: {e}")
        return None

# Create Users Table if not exists
def create_users_table():
    conn = create_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password VARCHAR(100) NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()

create_users_table()

# Password Hashing Functions
def hash_password(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def check_password(stored_password, entered_password):
    return bcrypt.checkpw(entered_password.encode("utf-8"), stored_password.encode("utf-8"))

# Sign Up Function
def sign_up(username, password, email):
    conn = create_connection()
    if conn:
        cursor = conn.cursor()
        hashed_password = hash_password(password)
        try:
            cursor.execute("INSERT INTO users (username, password, email) VALUES (%s, %s, %s)", (username, hashed_password, email))
            conn.commit()
            st.success("🎉 Sign Up Successful! Please log in.")
        except mysql.connector.Error as e:
            st.error(f"Error: {e}")
        finally:
            cursor.close()
            conn.close()

# Sign In Function
def sign_in(username, password):
    conn = create_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username, password FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        return user if user and check_password(user[1], password) else None

# Fetch Stock Data Function (With Caching)
def get_stock_data(symbol):
    current_time = time.time()
    if symbol in stock_cache and (current_time - stock_cache[symbol]["timestamp"] < 600):  
        return stock_cache[symbol]["data"]

    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={API_KEY}&outputsize=full"
    response = requests.get(url)
    data = response.json()

    stock_cache[symbol] = {"data": data, "timestamp": current_time}
    return data

# Process Stock Data
def process_stock_data(stock_data):
    if "Time Series (5min)" not in stock_data:
        st.error("⚠ No valid data received.")
        return None

    df = pd.DataFrame.from_dict(stock_data["Time Series (5min)"], orient="index")
    try:
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        df.fillna(method="ffill", inplace=True)
        return df
    except Exception as e:
        st.error(f"⚠ Data Processing Error: {e}")
        return None

# Stock Symbols
companies = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN"
}

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("", ["🏠 Home", "📊 Dashboard", "🚨 Alerts", "🔄 Compare Stocks"])

# Home Page (Sign In / Sign Up)
if page == "🏠 Home":
    st.title("📈 Stock Market Analyzer")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🔐 Sign In / Sign Up")
        tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

        with tab1:
            username = st.text_input("Username", key="signin_username")
            password = st.text_input("Password", type="password", key="signin_password")
            if st.button("Sign In"):
                if username and password:
                    user = sign_in(username, password)
                    if user:
                        st.session_state.user = user
                        st.success(f"Welcome back, {user[0]}!")
                    else:
                        st.error("Invalid username or password.")

        with tab2:
            new_username = st.text_input("Username", key="signup_username")
            new_password = st.text_input("Password", type="password", key="signup_password")
            email = st.text_input("Email", key="signup_email")
            if st.button("Sign Up"):
                if new_username and new_password and email:
                    sign_up(new_username, new_password, email)

# Stock Market Dashboard
elif page == "📊 Dashboard":
    st.title("📊 Stock Market Dashboard")
    selected_company = st.selectbox("📌 Select a Company", list(companies.keys()))

    if st.button("🔍 Fetch Stock Data"):
        stock_data = get_stock_data(companies[selected_company])

        if "Time Series (5min)" in stock_data:
            df = process_stock_data(stock_data)
            if df is not None:
                st.subheader(f"📈 {selected_company} Stock Details")
                fig = px.line(df, x=df.index, y="Close", title="📊 Intraday Stock Prices", template="plotly_dark")
                st.plotly_chart(fig)

# Stock Comparison
elif page == "🔄 Compare Stocks":
    st.title("🔄 Stock Comparison")

    stock1 = st.selectbox("📌 Select First Company", list(companies.keys()))
    stock2 = st.selectbox("📌 Select Second Company", [c for c in companies.keys() if c != stock1])

    if st.button("🔍 Compare Stocks"):
        stock1_data = process_stock_data(get_stock_data(companies[stock1]))
        stock2_data = process_stock_data(get_stock_data(companies[stock2]))

        if stock1_data is not None and stock2_data is not None:
            comparison_df = stock1_data[["Close"]].merge(stock2_data[["Close"]], left_index=True, right_index=True, suffixes=(f"_{stock1}", f"_{stock2}"))
            fig = px.line(comparison_df, title="📊 Price Comparison", template="plotly_dark")
            st.plotly_chart(fig)

# Price Alerts
elif page == "🚨 Alerts":
    st.title("🚨 Set Stock Price Alerts")
    selected_company = st.selectbox("📌 Choose a Company", list(companies.keys()))
    alert_price = st.number_input("💰 Enter Alert Price", min_value=0.0, format="%.2f")
    
    if st.button("✅ Set Alert"):
        st.success(f"🚀 Alert set for {selected_company} at ${alert_price:.2f}")
