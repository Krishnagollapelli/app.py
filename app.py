import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st

# Set Seaborn Style
sns.set_style("whitegrid")

# Fetch Real-Time COVID-19 Data (UK)
url = "https://disease.sh/v3/covid-19/countries/uk"
r = requests.get(url)
data = r.json()

# Extract relevant fields for analysis
covid_data = {
    "Total Cases": data["cases"],
    "Active Cases": data["active"],
    "Recovered": data["recovered"],
    "Deaths": data["deaths"],
}
df_covid = pd.DataFrame(list(covid_data.items()), columns=["Category", "Count"])

# Generate Random Historical Data (Last 30 Days)
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)  # Cases per day
df_historical = pd.DataFrame({"day": range(1, 31), "cases": historical_cases})

# Prepare Data for Regression Model
X = df_historical[["day"]]
y = df_historical["cases"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict Next 5 Days
future_days = np.array(range(1, 36)).reshape(-1, 1)
predicted_cases = model.predict(future_days)

# Streamlit App
st.title("COVID-19 Cases Prediction - UK")
st.write("Live COVID-19 Data and Future Predictions.")

# 📊 **Bar Graph 1: Real-Time COVID-19 Stats**
st.subheader("Current COVID-19 Statistics in the UK")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.barplot(x=df_covid["Category"], y=df_covid["Count"], palette="coolwarm", ax=ax1)
ax1.set_ylabel("Count")
ax1.set_title("COVID-19 Stats in UK (Live Data)")
st.pyplot(fig1)

# 📊 **Bar Graph 2: Historical vs. Predicted Cases**
st.subheader("Historical vs. Predicted Cases")
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.barplot(x=df_historical["day"], y=df_historical["cases"], color="blue", label="Actual Cases", ax=ax2)
sns.barplot(x=future_days.flatten(), y=predicted_cases, color="red", alpha=0.6, label="Predicted Cases", ax=ax2)
ax2.set_xlabel("Day")
ax2.set_ylabel("Cases")
ax2.set_title("Predicted vs. Actual COVID-19 Cases")
ax2.legend()
st.pyplot(fig2)

# 📈 **User Input for Prediction**
st.subheader("Predict COVID-19 Cases for Future Days")
day_input = st.number_input("Enter a day number for prediction (e.g., 31)", min_value=1, max_value=100)

if st.button("Predict Cases"):
    prediction = model.predict([[day_input]])
    st.write(f"Predicted cases for day {day_input}: **{int(prediction[0])}**")
