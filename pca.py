import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import streamlit as st

# Set Seaborn Style
sns.set_style("whitegrid")

# Generate synthetic historical data
np.random.seed(42)
days = np.arange(1, 31)
cases = np.random.randint(30000, 70000, size=30)
testing_rate = np.random.uniform(0.5, 1.0, size=30)
vaccination_rate = np.random.uniform(0.3, 0.9, size=30)
mobility_index = np.random.uniform(0.2, 0.8, size=30)

# Create DataFrame
df = pd.DataFrame({
    "day": days,
    "cases": cases,
    "testing_rate": testing_rate,
    "vaccination_rate": vaccination_rate,
    "mobility_index": mobility_index
})

# Features and Target
X = df[["day", "testing_rate", "vaccination_rate", "mobility_index"]]
y = df["cases"]

# Scaling Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA Transformation (Keep 2 components for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svm_model.fit(X_train, y_train)

# Prepare future prediction data (next 5 days)
future_days = np.arange(31, 36)
future_data = pd.DataFrame({
    "day": future_days,
    "testing_rate": np.random.uniform(0.5, 1.0, size=5),
    "vaccination_rate": np.random.uniform(0.3, 0.9, size=5),
    "mobility_index": np.random.uniform(0.2, 0.8, size=5)
})

# Scale and apply PCA to future data
future_scaled = scaler.transform(future_data)
future_pca = pca.transform(future_scaled)
predicted_cases = svm_model.predict(future_pca)

# ---------------------- Streamlit UI ------------------------
st.title("COVID-19 Cases Prediction with PCA + SVM")
st.write("Predicting COVID-19 cases using PCA for feature reduction and SVM regression.")

# 📊 Bar Chart - Historical Data
st.subheader("Historical COVID-19 Cases")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.barplot(x=df["day"], y=df["cases"], color="blue", ax=ax1)
ax1.set_xlabel("Day")
ax1.set_ylabel("Cases")
ax1.set_title("Last 30 Days COVID-19 Cases")
st.pyplot(fig1)

# 📊 Bar Chart - Predicted Future Cases
st.subheader("Predicted Future Cases (Next 5 Days)")
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.barplot(x=future_days, y=predicted_cases, color="orange", ax=ax2)
ax2.set_xlabel("Future Day")
ax2.set_ylabel("Predicted Cases")
ax2.set_title("SVM Prediction After PCA")
st.pyplot(fig2)

# 🧠 Prediction for Custom Input
st.subheader("Predict Cases for Custom Input")
col1, col2, col3, col4 = st.columns(4)

with col1:
    input_day = st.number_input("Day", min_value=1, max_value=100, value=31)
with col2:
    input_testing = st.slider("Testing Rate", 0.0, 1.0, 0.8)
with col3:
    input_vaccine = st.slider("Vaccination Rate", 0.0, 1.0, 0.5)
with col4:
    input_mobility = st.slider("Mobility Index", 0.0, 1.0, 0.6)

if st.button("Predict Custom Day"):
    user_input = pd.DataFrame([[
        input_day, input_testing, input_vaccine, input_mobility
    ]], columns=["day", "testing_rate", "vaccination_rate", "mobility_index"])
    
    user_scaled = scaler.transform(user_input)
    user_pca = pca.transform(user_scaled)
    user_prediction = svm_model.predict(user_pca)
    
    st.success(f"Predicted cases for day {input_day}: **{int(user_prediction[0])}**")

