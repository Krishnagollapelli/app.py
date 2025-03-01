import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Generate random historical data
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)  # Last 30 days cases
historical_deaths = np.random.randint(500, 2000, size=30)

df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = range(1, 31)

# Prepare data for SVM Regression
X = df_historical[["day"]]
y = df_historical["cases"]

# Scaling the features for SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# SVM Regression Model
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)

# Predict next day's cases
next_day = scaler.transform([[31]])
predicted_cases_svm = svm_model.predict(next_day)

# Prepare data for Logistic Regression (binary classification)
df_historical["high_cases"] = (df_historical["cases"] > 50000).astype(int)
X_classification = df_historical[["day"]]
y_classification = df_historical["high_cases"]

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

logistic_model = LogisticRegression()
logistic_model.fit(X_train_cls, y_train_cls)

# Streamlit App
st.title("COVID-19 Cases Prediction with SVM and Logistic Regression")
st.write("Predicting COVID-19 cases and classifying high/low case days.")

# User Input for Regression
st.subheader("SVM Regression Prediction")
day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

if st.button("Predict Cases with SVM"):
    day_scaled = scaler.transform([[day_input]])
    prediction = svm_model.predict(day_scaled)
    st.write(f"Predicted cases for day {day_input}: {int(prediction[0])}")

# User Input for Classification
st.subheader("Logistic Regression Classification")
day_input_cls = st.number_input("Enter day number for classification", min_value=1, max_value=30)

if st.button("Classify Day"):
    classification = logistic_model.predict([[day_input_cls]])
    category = "High Cases" if classification[0] == 1 else "Low Cases"
    st.write(f"Day {day_input_cls} is classified as: {category}")
