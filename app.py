import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Desertification Predictor", layout="centered")

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.title("🌍 Desertification Risk Prediction")
st.markdown("### AI-powered environmental risk analysis")

st.write("Enter climate values to predict risk")

# Inputs
rain = st.number_input("Rainfall", value=50.0)
temp = st.number_input("Temperature", value=30.0)
soil = st.number_input("Soil Moisture", value=0.5)

if st.button("Predict"):
    input_data = np.array([[rain, temp, soil]])
    pred = model.predict(input_data)

    if pred[0] == 2:
        st.error("🔥 High Desertification Risk")
    elif pred[0] == 1:
        st.warning("⚠️ Medium Risk")
    else:
        st.success("🌱 Low Risk")

    # 👇 MOVE GRAPH HERE
    import matplotlib.pyplot as plt

    features = ["Rain", "Temp", "Soil"]
    importance = model.feature_importances_

    fig, ax = plt.subplots()
    ax.bar(features, importance)
    st.pyplot(fig)