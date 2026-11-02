
import streamlit as st
import numpy as np
import pickle

# Load saved model
with open("svm_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

st.set_page_config(page_title="SVM Classifier", page_icon="🤖")

st.title("🤖 Support Vector Machine Classifier")
st.write("Predict the class using an SVM model.")

st.subheader("Enter Feature Values")

feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)

if st.button("Predict"):
    input_data = np.array([[feature1, feature2]])

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    st.success(f"Predicted Class: {prediction[0]}")

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(scaled_data)

        st.write("Prediction Probabilities:")
        st.write(probability[0])
