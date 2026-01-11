#pip install streamlit
import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Load saved objects (model, scaler, PCA)
# -----------------------------
model = joblib.load("/content/svm_wine_model.pkl")  # Best model (SVM)
scaler = joblib.load("/content/scaler.pkl")         # StandardScaler
pca = joblib.load("/content/pca.pkl")               # PCA transformer

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🍷 Wine Class Prediction App")
st.write("Enter the wine chemical properties to predict its class (0, 1, 2).")

# Wine dataset features
feature_names = [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
    'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
    'proanthocyanins', 'color_intensity', 'hue',
    'od280/od315_of_diluted_wines', 'proline'
]

# Input fields for all 13 features
user_input = []
for feature in feature_names:
    val = st.number_input(f"{feature}", value=0.0)
    user_input.append(val)

# Predict button
if st.button("Predict Wine Class"):
    # Convert input to numpy array
    input_array = np.array([user_input])

    # Scale input
    input_scaled = scaler.transform(input_array)

    # PCA transform
    input_pca = pca.transform(input_scaled)

    # Make prediction
    prediction = model.predict(input_pca)[0]

    st.success(f"Predicted Wine Class: {prediction}")
