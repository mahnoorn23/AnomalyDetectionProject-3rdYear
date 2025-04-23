import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# 1.) Page configuration
st.set_page_config(
    page_title="Home Water Monitoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2.) API Base URL
# Allow override via ENV var for dev vs. prod
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# 3.) Sidebar for model selection & threshold settings
st.sidebar.title("Settings")

model = st.sidebar.selectbox(
    "Select Model",
    ["iforest", "autoencoder"]
)

# Only display threshold slider for the autoencoder
if model == "autoencoder":
    st.sidebar.subheader("Autoencoder Threshold")
    new_thresh = st.sidebar.slider(
        "Threshold (L)",
        min_value=0.0,
        max_value=10.0,
        step=0.01,
        value=0.005
    )
    if st.sidebar.button("Save Threshold"):
        # Sends threshold to backend
        resp = requests.post(
            f"{API_BASE}/threshold",
            json={"model": "autoencoder", "threshold": new_thresh},
            timeout=5
        )
        if resp.ok:
            st.sidebar.success(f"Threshold set to {new_thresh} L")
        else:
            st.sidebar.error(f"Error: {resp.text}")

st.title("Domestic Water Monitoring Dashboard")

# 4.) Real-time anomaly check form
with st.form("predict_form", clear_on_submit=True):
    st.subheader("Real-Time Anomaly Check")
    flow = st.number_input(
        "Water Flow (L)",
        min_value=0.0,
        step=0.01,
        help="Enter current flow reading from your sensor"
    )
    email = st.text_input("Alert Email (optional)")
    phone = st.text_input("Alert Phone (optional)")
    submitted = st.form_submit_button("Check Anomaly")
    if submitted:
        try:
            payload = {"flow": flow}
            if email:
                payload["email"] = email
            if phone:
                payload["phone"] = phone
            res = requests.post(
                f"{API_BASE}/predict/{model}",
                json=payload,
                timeout=5
            )
            res.raise_for_status()
            data = res.json()
            status = "Anomaly Detected!" if data["anomaly"] else "Normal"
            st.metric("Status", status)
            st.write("Timestamp:", data["timestamp"])
            if model == "autoencoder":
                st.write("Threshold used:", data["threshold"])
        except Exception as e:
            st.error(f"Error calling API: {e}")

st.markdown("---")

#5.) Historical Usage
st.subheader("Historical Usage Data")
if st.button("Load History"):
    try:
        res = requests.get(f"{API_BASE}/history/{model}", timeout=5)
        res.raise_for_status()
        history = res.json()
        if not history:
            st.info("No data found yet.")
        else:
            df = pd.DataFrame(history)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # Line chart colored by the anomaly flag
            fig = px.line(
                df,
                x="timestamp",
                y="flow",
                color="anomaly",
                labels={"flow": "Water Flow (L)", "timestamp": "Time"},
                title="Water Flow Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error Fetching History: {e}")