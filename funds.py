import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
file_path = "effective_federal_funds_rate.csv"
df = pd.read_csv(file_path)

# Convert date to datetime and extract numerical feature
df['date'] = pd.to_datetime(df['date'])
df['date_ordinal'] = df['date'].map(lambda x: x.toordinal())

# Define features and target variable
X = df[['date_ordinal']]
y = df['value']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit App
st.title("Federal Funds Rate Prediction")

# Show dataset preview if checkbox is selected
if st.checkbox("Show Dataset"):
    st.write(df.head())

# Model Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

st.subheader("Model Evaluation")
st.write(f"Mean Absolute Error: {mae:.4f}")
st.write(f"Root Mean Squared Error: {rmse:.4f}")

# Plot actual vs predicted values
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(X_test, y_test, color='blue', label='Actual')
ax.scatter(X_test, y_pred, color='red', label='Predicted')
ax.set_xlabel("Date (Ordinal)")
ax.set_ylabel("Federal Funds Rate")
ax.legend()
st.pyplot(fig)

# Predict future values
def predict_future(date_str):
    date_ordinal = pd.to_datetime(date_str).toordinal()
    prediction = model.predict([[date_ordinal]])[0]
    return round(prediction, 4)

st.subheader("Predict Future Federal Funds Rate")
future_date = st.text_input("Enter a future date (YYYY-MM-DD):", "2025-06-01")
if st.button("Predict"):
    predicted_rate = predict_future(future_date)
    st.write(f"Predicted Federal Funds Rate for {future_date}: {predicted_rate}")
