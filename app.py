import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# Load saved model and preprocessing objects
model = joblib.load("predictive_maintenance_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Predictive Maintenance Dashboard", page_icon="⚙️", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1F3864; margin-bottom: 0px; }
    .sub-header { color: #666666; font-size: 1rem; margin-bottom: 1.5rem; }
    .stButton>button {
        background-color: #1F3864; color: white; font-weight: 600;
        border-radius: 8px; padding: 0.6rem 1.5rem; border: none;
    }
    .stButton>button:hover { background-color: #16294d; color: white; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">⚙️ Predictive Maintenance Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time equipment failure risk prediction using sensor data & machine learning</p>', unsafe_allow_html=True)
st.markdown("---")

left, right = st.columns([1, 1.2], gap="large")

with left:
    st.subheader("📥 Sensor Inputs")
    machine_type = st.selectbox("Machine Type", ["L", "M", "H"], help="L = Low, M = Medium, H = High quality variant")
    air_temp = st.slider("Air Temperature [K]", 295.0, 305.0, 298.0, 0.1)
    process_temp = st.slider("Process Temperature [K]", 305.0, 315.0, 308.0, 0.1)
    rot_speed = st.slider("Rotational Speed [rpm]", 1000, 2900, 1500)
    torque = st.slider("Torque [Nm]", 0.0, 80.0, 40.0, 0.5)
    tool_wear = st.slider("Tool Wear [min]", 0, 260, 100)

    predict_clicked = st.button("🔍 Predict Failure Risk", use_container_width=True)

with right:
    st.subheader("📊 Prediction Result")

    if predict_clicked:
        type_encoded = le.transform([machine_type])[0]
        features = np.array([[type_encoded, air_temp, process_temp, rot_speed, torque, tool_wear]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={'suffix': "%"},
            title={'text': "Failure Risk"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1F3864"},
                'steps': [
                    {'range': [0, 30], 'color': "#d4edda"},
                    {'range': [30, 70], 'color': "#fff3cd"},
                    {'range': [70, 100], 'color': "#f8d7da"},
                ],
            }
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        if prediction == 1:
            st.error(f"⚠️ **Failure Predicted** — {probability:.1%} probability")
        else:
            st.success(f"✅ **No Failure Predicted** — {probability:.1%} probability")

        with st.expander("ℹ️ What influences this prediction?"):
            st.write("""
            This model was trained on the **AI4I 2020 Predictive Maintenance Dataset**.
            Based on feature importance analysis, **Torque** and **Rotational Speed**
            are the strongest predictors of failure, followed by **Tool Wear**.
            """)
    else:
        st.info("👈 Set sensor values and click **Predict Failure Risk** to see results here.")

st.markdown("---")
st.caption("Model: Logistic Regression (class-balanced) | Dataset: AI4I 2020 Predictive Maintenance Dataset (UCI) | Built by Shalini")
