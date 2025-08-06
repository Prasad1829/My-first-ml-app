#importing requried modules
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import base64
import pandas as pd

st.set_page_config(layout="wide") #interface layout


#applying stylings for buttons and containers
st.markdown(
    """
<style>
.stheader {
    background: rgba(255, 255, 255, 0.1);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: #FF6347;
    font-family: 'Segoe UI', sans-serif;
    margin-bottom: 30px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.stheader h1 {
    font-size: 42px;
    margin-bottom: 10px;
    color: #FBE4D8;
}

.stheader p {
    font-size: 18px;
    color: #DFB6B2;
}

/* FORM ELEMENTS */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] {
    background-color: rgba(7, 41, 121, 0.356);
    border: 1px solid yellow;
    border-radius: 10px;
    color: white;
    padding: 10px;
}

/* Dropdown text color fix */
div[data-testid="stSelectbox"] label {
    color: #DFB6B2;
}

/* BUTTON STYLING */
.stButton>button {
    background: #FBE4D8; 
    backdrop-filter: blur(10px);
    color: #020100;
    font-size: 18px;
    border-radius: 20px;
    padding: 10px 20px;
    border: 1px solid rgb(0, 0, 0);
    transition: 0.3s;
}

.stButton>button:hover {
    background: #2B124C;
    color: #DFB6B2;
    transform: scale(1.05);
}

/* SALAAR THEME BACKGROUND */
.stApp {
    background: linear-gradient(
        160deg, 
        #0F0F0F 0%,     
        #1A1A1A 20%,    
        #2F3C4D 40%,    
        #4C5C68 65%,    
        #6A7D8C 85%,    
        #A0B5C6 100%    
    );
    min-height: 100vh;
    padding: 40px;
    font-family: 'Poppins', sans-serif;
}

/* GLASS EFFECT CONTAINER */
.blur-container {
    background: #d0d4debe; 
    backdrop-filter: blur(15px); 
    padding: 20px;
    border-radius: 10px;
    box-shadow: 10px 10px 10px #FFFFFF;
    margin-bottom: 20px;
    border: 10px solid rgba(255, 255, 255, 0.1);
}

/* OPTIONAL OVERLAY EFFECT */
.blur-overlay {
    backdrop-filter: blur(20px);
    background: rgba(255, 255, 255, 0.02);
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 30px;
}
.prediction-box {
 background: linear-gradient(135deg, #1a1a1a, #2b2b2b, #f0c420aa, #1a1a1a);
    color: #101010;
    border: 2px solid #39ff14;
    text-shadow: 1px 1px 3px #ffffff88;
    box-shadow: 0 0 12px #39ff1499;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    padding: 15px;
    margin-top: 20px;
    border-radius: 12px;
}
 
</style>
    """,
    unsafe_allow_html=True
)
st.markdown("""
<div class="header">
    <h1>Restaurant Rating Prediction App</h1>
    <p>This helps you to predict a restaurant review</p>
</div>
""", unsafe_allow_html=True)


#setting custom background 
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encode = base64.b64encode(image_file.read()).decode()
    background = f"""
    <style>
    .stApp{{
    background-image:url("data:image/png;base64,{encode}");
    background-size:cover;
    background-position:center;
    background-repeat:no-repeat;
    }}
    </style>
    """
    st.markdown(background, unsafe_allow_html=True)

#loading models

scaler = joblib.load("Restaurant_rating.pkl")

model = joblib.load("Rating_predict.pkl")


st.divider()

# Input fields
average_cost = st.number_input(" 👥 Average Cost for Two", min_value=50, max_value=999999999, value=500)
table_booking = st.selectbox(" 🥢Table Booking Available?", ["Yes", "No"])
online_delivery = st.selectbox(" 🚚 Online Delivery Available?", ["Yes", "No"])
price_range = st.selectbox(" 💸Price Range (1-4)", [1, 2, 3, 4])


model = joblib.load("Rating_predict.pkl")

#converting "yes/no" to binary 
# Convert inputs once and prepare scaled features
booking_status = 1 if table_booking == "Yes" else 0
delivery_status = 1 if online_delivery == "Yes" else 0
feature_names = ['Average Cost for two', 'Has Table booking', 'Has Online delivery', 'Price range']
input_df = pd.DataFrame([[average_cost, booking_status, delivery_status, price_range]], columns=feature_names)
scaled_features = scaler.transform(input_df)

# Button
predictbutton = st.button("Review")
st.divider()

if predictbutton:
    st.balloons()# applying now effect

    # categorizing the predicting rating
    prediction = model.predict(scaled_features)
    if prediction < 2.5:
        st.markdown('<div class="prediction-box bad">Bad</div>', unsafe_allow_html=True)
    elif prediction < 3.5:
        st.markdown('<div class="prediction-box avg">Average</div>', unsafe_allow_html=True)
    elif prediction < 4.0:
        st.markdown('<div class="prediction-box good">Good</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="prediction-box excellent">Excellent</div>', unsafe_allow_html=True)

