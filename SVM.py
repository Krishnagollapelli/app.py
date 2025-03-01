import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Set Seaborn Style
sns.set_style("whitegrid")

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

# Train the SVM Regression Model
svm_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svm_model.fit(X_train, y_train)

# Predict cases for visualization
days_future = np.array(range(1, 35)).reshape(-1, 1)  # Predict for 4 extra days
days_future_scaled = scaler.transform(days_future)
predicted_cases_svm = svm_model.predict(days_future_scaled)

# Streamlit App
st.title("COVID-19 Cases Prediction in USA")
st.write("Advanced prediction of COVID-19 cases using Machine Learning.")

# 📊 **Bar Graph 1: Historical COVID-19 Cases**
st.subheader("Historical COVID-19 Cases")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.barplot(x=df_historical["day"], y=df_historical["cases"], color="blue", ax=ax1)
ax1.set_xlabel("Day")
ax1.set_ylabel("Number of Cases")
ax1.set_title("Past 30 Days COVID-19 Cases")
st.pyplot(fig1)

# 📊 **Bar Graph 2: SVM Predicted Cases vs. Actual Cases**
st.subheader("SVM Predicted Cases vs. Actual Cases")
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.barplot(x=df_historical["day"], y=df_historical["cases"], color="blue", label="Actual Cases", ax=ax2)
sns.barplot(x=days_future.flatten(), y=predicted_cases_svm, color="red", alpha=0.6, label="Predicted Cases", ax=ax2)
ax2.set_xlabel("Day")
ax2.set_ylabel("Number of Cases")
ax2.set_title("SVM Regression Prediction of COVID-19 Cases")
ax2.legend()
st.pyplot(fig2)

# 📈 **SVM Regression Prediction for User Input**
st.subheader("Predict COVID-19 Cases Using SVM")
day_input = st.number_input("Enter day number for prediction (e.g., 31)", min_value=1, max_value=100)

if st.button("Predict Cases with SVM"):
    day_scaled = scaler.transform([[day_input]])
    prediction = svm_model.predict(day_scaled)
    st.write(f"Predicted cases for day {day_input}: **{int(prediction[0])}**")


