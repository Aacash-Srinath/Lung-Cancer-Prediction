# Import Dependencies
import streamlit as st
import numpy as np
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Page Configurations
st.set_page_config(page_title="Lung Cancer Prediction | Aacash Srinath",
                   page_icon=None, layout="wide",
                   initial_sidebar_state="collapsed",
                   menu_items=None)

# Page Headings
st.title("Lung Cancer Prediction App")
st.subheader("Web App for Predicting if the Person with Given Symptoms has Lung Cancer or not")
st.write('')

# Loading Model
LogReg = joblib.load('LCD_LogReg.pkl')

# Instructions for Filling the Fields
st.write("0=NO, 1=YES")

# Input Fields for Symptoms
col1, col2, col3 = st.columns([1,1,1])
with col1:
    yf = st.selectbox("Do you have Yellow Fingers?", [0, 1])
    anxiety = st.selectbox("Do you have Anxiety", [0, 1])
    peerpress = st.selectbox("Do you face Peer Pressure to Smoke?", [0, 1])
    chronicdisease = st.selectbox("Do you have any Chronic Diseases?", [0, 1])
with col2:
    fatigue = st.selectbox("Are you Fatigued most of the time?", [0, 1])
    allergy = st.selectbox("Do you have any Allergies?", [0, 1])
    wheezing = st.selectbox("Do you have Wheezing?", [0, 1])
with col3:
    alcohol = st.selectbox("Do you drink Alcohol?", [0, 1])
    coughing = st.selectbox("Do you Cough frequently?", [0, 1])
    swallowing = st.selectbox("Do you face Difficulty in Swallowing?", [0, 1])
    chestpain = st.selectbox("Do you have any Chest Pain?", [0, 1])

# Creating Dataframe of Inputs
data = {"YELLOW_FINGERS":[yf], "ANXIETY":[anxiety], "PEER_PRESSURE":[peerpress], "CHRONIC DISEASE":[chronicdisease], "FATIGUE ":[fatigue], "ALLERGY ":[allergy], "WHEEZING":[wheezing], "ALCOHOL CONSUMING":[alcohol], "COUGHING":[coughing], "SWALLOWING DIFFICULTY":[swallowing], "CHEST PAIN":[chestpain]}
cancer = pd.DataFrame.from_dict(data)
cancer["ANXYELFIN"] = (cancer["ANXIETY"]) * (cancer["YELLOW_FINGERS"])

# Prediction for the Given Values
predicted = LogReg.predict(cancer)
st.write("")
st.write("")

# Button for Viewing the Predition
view_pred = False
view_pred = st.button("Predict")
if (view_pred):
    if (predicted==1):
        st.subheader("We are sorry to inform you that you have possibile symptoms of Lung Cancer")
    else:
        st.subheader("Congratulations, you currently show no symptoms of Lung Cancer!!")