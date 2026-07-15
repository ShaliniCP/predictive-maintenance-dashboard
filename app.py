import streamlit as st
import joblib
import numpy as np

# Load saved model and preprocessing objects
model = joblib.load("predictive_maintenance_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Predictive Maintenance Dashboard", page_icon="⚙️")

st.title("⚙️ Predictive Maintenance Dashboard")
st.write("Enter machine sensor readings below to predict failure risk.")

# Input fields
machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
air_temp = st.number_input("Air Temperature [K]", value=298.0)
process_temp = st.number_input("Process Temperature [K]", value=308.0)
rot_speed = st.number_input("Rotational Speed [rpm]", value=1500)
torque = st.number_input("Torque [Nm]", value=40.0)
tool_wear = st.number_input("Tool Wear [min]", value=100)

if st.button("Predict Failure Risk"):
    type_encoded = le.transform([machine_type])[0]

    features = np.array([[type_encoded, air_temp, process_temp, rot_speed, torque, tool_wear]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"⚠️ Failure Predicted — Probability: {probability:.2%}")
    else:
        st.success(f"✅ No Failure Predicted — Probability: {probability:.2%}")

    st.progress(float(probability))

st.markdown("---")
st.caption("Model: Logistic Regression | Dataset: AI4I 2020 Predictive Maintenance Dataset (UCI)")