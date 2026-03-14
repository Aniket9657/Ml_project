import streamlit as st
import joblib
import numpy as np

model = joblib.load("driver_risk_model.pkl")

st.title("🚗 Driver Risk Prediction System")

st.sidebar.header("Driver Information")

driver_age = st.sidebar.slider("Driver Age",18,70,30)
experience = st.sidebar.slider("Experience (years)",0,40,8)

avg_speed = st.sidebar.slider("Average Speed",40,140,90)
max_speed = st.sidebar.slider("Max Speed",50,160,110)

lane_changes = st.sidebar.slider("Lane Changes",0,15,5)
acceleration_rate = st.sidebar.slider("Acceleration Rate",1.0,6.0,3.0)
harsh_brakes = st.sidebar.slider("Harsh Brakes",0,6,1)

following_distance = st.sidebar.slider("Following Distance",5,30,10)
speed_variation = st.sidebar.slider("Speed Variation",1,20,8)

car_type = 1
locality = 0
profession = 2
previous_accidents = 1
traffic_violations = 2

if st.button("Predict Risk"):

    data = np.array([[driver_age,experience,car_type,locality,profession,
    previous_accidents,traffic_violations,avg_speed,max_speed,lane_changes,
    acceleration_rate,harsh_brakes,following_distance,speed_variation]])

    prediction = model.predict(data)

    risk_levels = ["Safe","Moderate Risk","High Risk"]

    st.subheader("Prediction Result")

    if prediction[0] == 2:
        st.error("⚠ High Risk Driver Detected")
    elif prediction[0] == 1:
        st.warning("Moderate Risk Driver")
    else:
        st.success("Safe Driver")