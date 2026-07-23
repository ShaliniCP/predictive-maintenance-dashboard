import streamlit as st
import joblib
import numpy as np

model = joblib.load("predictive_maintenance_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Predictive Maintenance Dashboard", page_icon="⚙️", layout="wide")

# ----------------------------------------------------------------------------
# THEME / CSS
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500&display=swap');

:root {
    --bg-primary: #14181C;
    --bg-panel: #1C2228;
    --bg-panel-alt: #232A31;
    --border-col: #2E363E;
    --accent-amber: #F2A93B;
    --accent-teal: #45C4B0;
    --accent-red: #E5484D;
    --text-primary: #E8EAED;
    --text-muted: #8B95A1;
}

.stApp {
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* Recolor Streamlit's default top header/toolbar to match dark theme */
[data-testid="stHeader"] {
    background-color: var(--bg-primary) !important;
}
[data-testid="stToolbar"] {
    background-color: var(--bg-primary) !important;
}
[data-testid="stDecoration"] {
    background-image: none !important;
    background-color: var(--bg-primary) !important;
}

/* Kill default Streamlit padding blocks a bit */
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* ---------- Header / Nameplate ---------- */
.plate-header {
    border: 1px solid var(--border-col);
    background: linear-gradient(180deg, var(--bg-panel-alt) 0%, var(--bg-panel) 100%);
    border-left: 4px solid var(--accent-amber);
    padding: 20px 28px;
    margin-bottom: 28px;
    border-radius: 2px;
}
.plate-eyebrow {
    font-family: 'Oswald', sans-serif;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 3px;
    color: var(--accent-amber);
    text-transform: uppercase;
    margin: 0 0 6px 0;
}
.plate-title {
    font-family: 'Oswald', sans-serif;
    font-size: 30px;
    font-weight: 700;
    letter-spacing: 1px;
    color: var(--text-primary);
    text-transform: uppercase;
    margin: 0;
}
.plate-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    color: var(--text-muted);
    margin-top: 8px;
    letter-spacing: 0.5px;
}

/* ---------- Section labels (nameplate style) ---------- */
.section-label {
    font-family: 'Oswald', sans-serif;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border-col);
    padding-bottom: 8px;
    margin-bottom: 18px;
}
.section-label span {
    color: var(--accent-amber);
}

/* ---------- Panel wrapper (native st.container border) ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-panel);
    border: 1px solid var(--border-col) !important;
    border-radius: 3px !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
    background: var(--bg-panel);
}

/* ---------- Sensor row w/ LED ---------- */
.sensor-row-label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 13px;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 2px;
}
.led {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    background: var(--accent-teal);
    box-shadow: 0 0 6px var(--accent-teal);
}

/* ---------- Streamlit widget overrides ---------- */
[data-testid="stSlider"] label {
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: var(--text-muted) !important;
    font-size: 13px !important;
}
[data-testid="stSlider"] {
    padding-bottom: 6px;
}
div[data-baseweb="select"] > div {
    background-color: var(--bg-panel-alt) !important;
    border-color: var(--border-col) !important;
    color: var(--text-primary) !important;
}
/* Force amber on any inline-styled element inside the slider (covers both hex and rgb formats
   Streamlit may use for the filled track + thumb), while leaving the unfilled track untouched. */
[data-testid="stSlider"] div[data-baseweb="slider"] div[style*="background-color"] {
    background-color: var(--accent-amber) !important;
    border-color: var(--accent-amber) !important;
}
[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
    background-color: var(--accent-amber) !important;
    border-color: var(--accent-amber) !important;
    box-shadow: none !important;
}
/* The current-value label shown above the slider handle also inherits theme color inline */
[data-testid="stSlider"] div[style*="color"] {
    color: var(--accent-amber) !important;
}
[data-testid="stThumbValue"] {
    color: var(--accent-amber) !important;
}

/* Button */
.stButton>button {
    background-color: var(--accent-amber) !important;
    color: #14181C !important;
    font-family: 'Oswald', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase;
    font-size: 13px !important;
    border-radius: 2px !important;
    border: none !important;
    padding: 0.65rem 1.4rem !important;
    width: 100%;
}
.stButton>button:hover {
    background-color: #ffbe5c !important;
    color: #14181C !important;
}

/* ---------- Readout ---------- */
.readout-wrap {
    text-align: center;
    padding: 10px 0 4px 0;
}
.readout-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 56px;
    font-weight: 600;
    line-height: 1;
}
.readout-caption {
    font-family: 'Oswald', sans-serif;
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 6px;
}

/* ---------- Custom meter gauge ---------- */
.meter-wrap { margin: 18px 0 6px 0; }
.meter-track {
    position: relative;
    height: 14px;
    border-radius: 2px;
    background: linear-gradient(90deg,
        var(--accent-teal) 0%, var(--accent-teal) 30%,
        var(--accent-amber) 30%, var(--accent-amber) 70%,
        var(--accent-red) 70%, var(--accent-red) 100%);
    border: 1px solid var(--border-col);
    overflow: visible;
}
.meter-ticks {
    display: flex;
    justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 4px;
}
.meter-needle {
    position: absolute;
    top: -6px;
    width: 3px;
    height: 26px;
    background: #F5F7FA;
    box-shadow: 0 0 4px rgba(255,255,255,0.6);
    transform: translateX(-50%);
}

/* ---------- Status banner ---------- */
.status-banner {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14px;
    font-weight: 500;
    padding: 14px 18px;
    border-radius: 2px;
    margin-top: 18px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.status-ok {
    background: rgba(69, 196, 176, 0.12);
    border: 1px solid var(--accent-teal);
    color: var(--accent-teal);
}
.status-fail {
    background: rgba(229, 72, 77, 0.12);
    border: 1px solid var(--accent-red);
    color: var(--accent-red);
}

/* ---------- Feature bars ---------- */
.feat-row { margin-bottom: 10px; }
.feat-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    color: var(--text-muted);
    display: flex;
    justify-content: space-between;
    margin-bottom: 3px;
}
.feat-bar-bg {
    background: var(--bg-panel-alt);
    border: 1px solid var(--border-col);
    height: 6px;
    border-radius: 2px;
}
.feat-bar-fill {
    background: var(--accent-amber);
    height: 100%;
    border-radius: 2px;
}

.footer-note {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--text-muted);
    text-align: center;
    margin-top: 32px;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown("""
<div class="plate-header">
    <div class="plate-eyebrow">Unit Monitoring &nbsp;/&nbsp; Predictive Diagnostics</div>
    <div class="plate-title">Predictive Maintenance Dashboard</div>
    <div class="plate-sub">MODEL: LOGREG-CLF-01 &nbsp;|&nbsp; DATASET: AI4I-2020 &nbsp;|&nbsp; STATUS: ONLINE</div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1.15], gap="large")

FEATURE_IMPORTANCE = {
    "Torque [Nm]": 0.319,
    "Rotational Speed [rpm]": 0.291,
    "Tool Wear [min]": 0.209,
    "Air Temperature [K]": 0.104,
    "Process Temperature [K]": 0.063,
}

with left:
    with st.container(border=True):
        st.markdown('<div class="section-label"><span>01</span> &nbsp;Sensor Input Panel</div>', unsafe_allow_html=True)

        machine_type = st.selectbox("Machine Type", ["L", "M", "H"], help="L = Low, M = Medium, H = High quality variant")
        air_temp = st.slider("Air Temperature [K]", 295.0, 305.0, 298.0, 0.1)
        process_temp = st.slider("Process Temperature [K]", 305.0, 315.0, 308.0, 0.1)
        rot_speed = st.slider("Rotational Speed [rpm]", 1000, 2900, 1500)
        torque = st.slider("Torque [Nm]", 0.0, 80.0, 40.0, 0.5)
        tool_wear = st.slider("Tool Wear [min]", 0, 260, 100)

        predict_clicked = st.button("Run Diagnostic")

with right:
    with st.container(border=True):
        st.markdown('<div class="section-label"><span>02</span> &nbsp;Diagnostic Output</div>', unsafe_allow_html=True)

        if predict_clicked:
            type_encoded = le.transform([machine_type])[0]
            features = np.array([[type_encoded, air_temp, process_temp, rot_speed, torque, tool_wear]])
            features_scaled = scaler.transform(features)

            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1]
            pct = probability * 100

            color = "var(--accent-teal)" if pct < 30 else ("var(--accent-amber)" if pct < 70 else "var(--accent-red)")

            st.markdown(f"""
            <div class="readout-wrap">
                <div class="readout-value" style="color:{color};">{pct:05.1f}%</div>
                <div class="readout-caption">Failure Risk Probability</div>
            </div>
            <div class="meter-wrap">
                <div class="meter-track">
                    <div class="meter-needle" style="left:{min(max(pct,1),99)}%;"></div>
                </div>
                <div class="meter-ticks"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div>
            </div>
            """, unsafe_allow_html=True)

            if prediction == 1:
                st.markdown(f'<div class="status-banner status-fail">⚠ FAILURE PREDICTED — recommend inspection before continued operation</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-banner status-ok">✓ NO FAILURE PREDICTED — unit operating within learned normal range</div>', unsafe_allow_html=True)

            st.markdown('<div style="height:22px;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-label" style="font-size:11px; border:none; padding-bottom:0;">Model Feature Weighting</div>', unsafe_allow_html=True)
            for feat, val in FEATURE_IMPORTANCE.items():
                width = int(val / max(FEATURE_IMPORTANCE.values()) * 100)
                st.markdown(f"""
                <div class="feat-row">
                    <div class="feat-label"><span>{feat}</span><span>{val:.3f}</span></div>
                    <div class="feat-bar-bg"><div class="feat-bar-fill" style="width:{width}%;"></div></div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center; padding: 60px 10px; color: var(--text-muted); font-family:'IBM Plex Sans',sans-serif; font-size:13px;">
                Awaiting input &nbsp;—&nbsp; set sensor values on the left panel<br>and select <b style="color:var(--accent-amber);">RUN DIAGNOSTIC</b> to view results.
            </div>
            """, unsafe_allow_html=True)

st.markdown('<div class="footer-note">LOGISTIC REGRESSION (CLASS-BALANCED) &nbsp;·&nbsp; AI4I 2020 PREDICTIVE MAINTENANCE DATASET (UCI) &nbsp;·&nbsp; BUILT BY SHALINI</div>', unsafe_allow_html=True)
