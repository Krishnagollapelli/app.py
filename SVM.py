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

# Advanced SVM Regression Model
svm_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svm_model.fit(X_train, y_train)

# Predict cases for visualization
days_future = np.array(range(1, 35)).reshape(-1, 1)  # Predict for 4 extra days
days_future_scaled = scaler.transform(days_future)
predicted_cases_svm = svm_model.predict(days_future_scaled)

# Prepare data for Logistic Regression (binary classification)
df_historical["high_cases"] = (df_historical["cases"] > 50000).astype(int)
X_classification = df_historical[["day"]]
y_classification = df_historical["high_cases"]

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# Logistic Regression Model
logistic_model = LogisticRegression(C=1.0, solver='liblinear')
logistic_model.fit(X_train_cls, y_train_cls)

# Streamlit App
st.title("COVID-19 Cases Prediction in USA")
st.write("Advanced prediction and classification of COVID-19 cases.")

### 📊 **Historical Data Graph**
st.subheader("Historical COVID-19 Cases")
fig1, ax1 = plt.subplots()
ax1.plot(df_historical["day"], df_historical["cases"], label="Actual Cases", marker="o", linestyle="-", color="blue")
ax1.set_xlabel("Day")
ax1.set_ylabel("Number of Cases")
ax1.set_title("Past 30 Days COVID-19 Cases")
ax1.legend()
st.pyplot(fig1)

### 📈 **SVM Regression Graph**
st.subheader("SVM Predicted Cases vs. Actual Cases")
fig2, ax2 = plt.subplots()
ax2.scatter(df_historical["day"], df_historical["cases"], label="Actual Cases", color="blue")
ax2.plot(days_future, predicted_cases_svm, label="SVM Predicted Cases", linestyle="--", color="red")
ax2.set_xlabel("Day")
ax2.set_ylabel("Number of Cases")
ax2.set_title("SVM Regression Prediction of COVID-19 Cases")
ax2.legend()
st.pyplot(fig2)

# User Input for Regression
st.subheader("Advanced SVM Regression Prediction")
day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

if st.button("Predict Cases with SVM"):
    day_scaled = scaler.transform([[day_input]])
    prediction = svm_model.predict(day_scaled)
    st.write(f"Predicted cases for day {day_input}: {int(prediction[0])}")

# User Input for Classification
st.subheader("Advanced Logistic Regression Classification")
day_input_cls = st.number_input("Enter day number for classification", min_value=1, max_value=30)

if st.button("Classify Day"):
    classification = logistic_model.predict([[day_input_cls]])
    category = "High Cases" if classification[0] == 1 else "Low Cases"
    st.write(f"Day {day_input_cls} is classified as: {category}")
