import os

import joblib
import pandas as pd
import streamlit as st

st.title("House Rent Prediction")

base_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_path = os.path.join(base_dir, "house_rent_pipeline.pkl")
pipeline = joblib.load(pipeline_path)

BHK = st.number_input("Enter the number of BHK:", min_value=1, max_value=10, step=1)
Size = st.number_input("Enter the size of the house in square feet:", min_value=100, max_value=10000, step=10)
Area_Type = st.selectbox("Select the area type:", ["Super Area", "Carpet Area", "Built Area"])
City = st.selectbox("Select the city:", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata"])
Furnishing_Status = st.selectbox("Select the furnishing status:", ["Furnished", "Semi-Furnished", "Unfurnished"])
Tenant_Preferred = st.selectbox("Select the tenant preferred:", ["Family", "Bachelors", "Any"])
Bathroom = st.number_input("Enter the number of bathrooms:", min_value=1, max_value=10, step=1)
Point_of_Contact = st.selectbox("Select the point of contact:", ["Contact Owner", "Contact Agent", "Contact Builder"])

input_df = pd.DataFrame(
    {
        "BHK": [BHK],
        "Size": [Size],
        "Area Type": [Area_Type],
        "City": [City],
        "Furnishing Status": [Furnishing_Status],
        "Tenant Preferred": [Tenant_Preferred],
        "Bathroom": [Bathroom],
        "Point of Contact": [Point_of_Contact],
    }
)

if st.button("Predict Rent"):
    prediction = pipeline.predict(input_df)[0]
    st.write(f"The predicted rent is: {prediction}")