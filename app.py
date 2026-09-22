import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# Title & Description
st.title("🩺 Breast Cancer Risk Detection")
st.markdown(
    """
    Kripya neeche diye gaye 5 mukhya (important) clinical parameters enter karein.
    Yeh tool Neural Network model ka upayog karke Cancer Risk ka anumaan lagata hai.
    """
)

# Model Training & Caching (Top 5 Features par train kiya gaya model)
@st.cache_resource
def train_simple_model():
    data = load_breast_cancer()
    # Selected top 5 features: mean radius (0), mean texture (1), mean perimeter (2), mean area (3), mean concavity (6)
    selected_indices = [0, 1, 2, 3, 6]
    X = data.data[:, selected_indices]
    y = data.target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Simple Neural Network Build
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=80, batch_size=16, verbose=0)

    return model, scaler

model, scaler = train_simple_model()

st.divider()

# Input UI Controls (5 Important Inputs Only)
st.subheader("📋 Patient Clinical Features")

col1, col2 = st.columns(2)

with col1:
    mean_radius = st.number_input(
        "1. Mean Radius",
        min_value=6.0, max_value=30.0, value=14.12, step=0.1,
        help="Tumor perimeter se center tak ki ausat doori (mm)"
    )
    mean_texture = st.number_input(
        "2. Mean Texture",
        min_value=9.0, max_value=40.0, value=19.28, step=0.1,
        help="Gray-scale values ka standard deviation"
    )
    mean_perimeter = st.number_input(
        "3. Mean Perimeter",
        min_value=40.0, max_value=190.0, value=91.96, step=0.1,
        help="Tumor ka outer boundary measurement"
    )

with col2:
    mean_area = st.number_input(
        "4. Mean Area",
        min_value=140.0, max_value=2500.0, value=654.8, step=1.0,
        help="Tumor ka kul kshetraphal (sq mm)"
    )
    mean_concavity = st.number_input(
        "5. Mean Concavity",
        min_value=0.0, max_value=0.5, value=0.088, step=0.001, format="%.4f",
        help="Contour ke concave bhaag ki severity"
    )

st.divider()

# Predict Button
if st.button("📊 Analyze & Predict Risk", type="primary", use_container_width=True):
    # Process Inputs
    user_data = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_concavity]])
    user_data_scaled = scaler.transform(user_data)
    
    # Model Prediction
    prediction_prob = model.predict(user_data_scaled)[0][0]
    
    st.subheader("🔍 Prediction Results")
    
    # Benign vs Malignant Logic
    if prediction_prob >= 0.5:
        st.success("✅ **Result: Benign (Non-Cancerous)**")
        st.write(f"**Confidence Score:** {prediction_prob * 100:.2f}% (Normal/Safe Tissue)")
        st.progress(float(prediction_prob))
    else:
        st.error("⚠️ **Result: Malignant (Cancerous)**")
        st.write(f"**Cancer Risk Score:** {(1 - prediction_prob) * 100:.2f}%")
        st.progress(float(1 - prediction_prob))

    st.caption("Note: Yeh prediction kewal educational purpose ke liye hai. Medical decision ke liye doctor se sampark karein.")