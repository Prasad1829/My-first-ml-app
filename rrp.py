#importing requried modules
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import base64

st.set_page_config(layout="wide") #interface layout


#applying stylings for buttons and containers
st.markdown(
    """
    <style>

    .blur-container {
        background: rgba(255, 255, 255, 0.05); 
        backdrop-filter: blur(15px); 
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.5);
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stTextInput, .stNumberInput, .stSelectbox {
        color: white !important;
        background: rgba(0, 0, 255, 0.1) !important; 
        backdrop-filter: blur(10px) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(0, 0, 255, 0.1) !important;
        padding: 8px !important;
    }

   
    .stButton>button {
        background: rgba(0, 255, 0, 0.0); 
        backdrop-filter: blur(10px);
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 20px;
        border: 1px solid rgba(0, 0, 0, 0.0);
        transition: 0.3s;
    }

    .stButton>button:hover {
        background: rgba(0, 0, 0, 0.0);
    }
    .stApp {
        padding: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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

#reading background image
'''set_background(r"C:\Users\kusar\OneDrive\Pictures\abstract-colorful-7-4k.jpg")'''

#loading models

scaler = joblib.load("Restaurant_rating.pkl")

model = joblib.load("Rating_predict.pkl")

# title
st.title("Restaurant Rating Prediction App")

#description
st.caption("This Help's you to predict a Restaurant review")

st.divider()

# Input fields
average_cost = st.number_input(" 👥 Average Cost for Two", min_value=50, max_value=999999999, value=500)
table_booking = st.selectbox(" 🥢Table Booking Available?", ["Yes", "No"])
online_delivery = st.selectbox(" 🚚 Online Delivery Available?", ["Yes", "No"])
price_range = st.selectbox(" 💸Price Range (1-4)", [1, 2, 3, 4])
# Convert inputs
booking_status = 1 if table_booking == "Yes" else 0
delivery_status = 1 if online_delivery == "Yes" else 0
input_data = np.array([[average_cost, booking_status, delivery_status, price_range]])
input_scaled = scaler.transform(input_data)

predictbutton = st.button("Predict the Review!")

st.divider()

model = joblib.load("Rating_predict.pkl")

#converting "yes/no" to binary 
bookingstatus = 1 if table_booking == "Yes" else 0 

deliverystatus =  1 if online_delivery == "Yes" else 0

values = [[average_cost,bookingstatus,deliverystatus,price_range]]

my_X_values = np.array(values)  #converting list to array

X = scaler.transform(my_X_values) #scaling input features
 
if predictbutton:
    st.snow() # applying now effect

#categorzing the predicting rating
    prediction = model.predict(X)

    if prediction < 2.5:
        st.write("Bad")
    elif prediction < 3.5:
        st.write ("Average")
    elif prediction < 4.0:
        st.write("Good")
    else:
        st.write("Excellent")