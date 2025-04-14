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

# ---------------------- Data Generation ------------------------
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

# ---------------------- Preprocessing ------------------------
X = df[["day", "testing_rate", "vaccination_rate", "mobility_index"]]
y = df["cases"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# PCA insights
explained_variance = pca.explained_variance_ratio_
components_df = pd.DataFrame(pca.components_, columns=X.columns, index=['PC1', 'PC2'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# SVM Regression Model
svm_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svm_model.fit(X_train, y_train)

# ---------------------- Future Prediction ------------------------
future_days = np.arange(31, 36)
future_data = pd.DataFrame({
    "day": future_days,
    "testing_rate": np.random.uniform(0.5, 1.0, size=5),
    "vaccination_rate": np.random.uniform(0.3, 0.9, size=5),
    "mobility_index": np.random.uniform(0.2, 0.8, size=5)
})

future_scaled = scaler.transform(future_data)
future_pca = pca.transform(future_scaled)
predicted_cases = svm_model.predict(future_pca)

# ---------------------- Streamlit UI ------------------------
st.title("🦠 COVID-19 Case Prediction with PCA + SVM")
st.markdown("This app uses **Principal Component Analysis (PCA)** to reduce features, then predicts future COVID-19 cases using **Support Vector Regression (SVM)**.")

# 📊 Historical Data Chart
st.subheader("📊 Historical COVID-19 Cases (Last 30 Days)")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.barplot(x=df["day"], y=df["cases"], color="blue", ax=ax1)
ax1.set_xlabel("Day")
ax1.set_ylabel("Cases")
ax1.set_title("COVID-19 Cases - Past 30 Days")
st.pyplot(fig1)

# 📈 Predicted Cases for Future
st.subheader("📈 Predicted COVID-19 Cases (Next 5 Days)")
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.barplot(x=future_days, y=predicted_cases, color="orange", ax=ax2)
ax2.set_xlabel("Future Day")
ax2.set_ylabel("Predicted Cases")
ax2.set_title("SVM Regression Prediction")
st.pyplot(fig2)

# 📉 PCA Analysis
st.subheader("📉 PCA Analysis (Dimensionality Reduction)")

# Explained Variance Ratio Plot
st.markdown("**Explained Variance Ratio:** Shows how much information each principal component retains.")
fig_var, ax_var = plt.subplots()
sns.barplot(x=['PC1', 'PC2'], y=explained_variance, palette="viridis", ax=ax_var)
ax_var.set_ylabel("Variance Explained")
ax_var.set_ylim(0, 1)
st.pyplot(fig_var)

# Feature Contribution Table
st.markdown("**Feature Contribution to Principal Components:**")
st.dataframe(components_df.style.background_gradient(cmap='coolwarm').format("{:.2f}"))

# PCA Scatter Plot
st.markdown("**PCA Scatter Plot (Reduced 2D Representation of Data):**")
fig_scatter, ax_scatter = plt.subplots(figsize=(7, 5))
scatter = ax_scatter.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", s=80)
ax_scatter.set_xlabel("Principal Component 1")
ax_scatter.set_ylabel("Principal Component 2")
ax_scatter.set_title("COVID-19 Data in PCA-Reduced Space")
fig_scatter.colorbar(scatter, ax=ax_scatter, label='Cases')
st.pyplot(fig_scatter)

# 🧠 Custom Prediction
st.subheader("🎯 Predict Custom Day")
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
    user_input = pd.DataFrame([[input_day, input_testing, input_vaccine, input_mobility]],
                              columns=["day", "testing_rate", "vaccination_rate", "mobility_index"])
    user_scaled = scaler.transform(user_input)
    user_pca = pca.transform(user_scaled)
    user_prediction = svm_model.predict(user_pca)
    st.success(f"Predicted cases for day {input_day}: **{int(user_prediction[0])}**")
