import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_dispatcher import models
from src.data_preprocessing import preprocess_data

from src.data_cleaning import drop_columns, impute_missing_values
from src.data_preprocessing import preprocess_data
from src.model_utils import evaluate_model

# Load the model and imputer
model = joblib.load("./models/log_reg_fold_0.pkl")
mode_imputer = joblib.load("./models/imputer_fold_0.pkl")


# Define the selected features based on your model
selected_features = [
    'Do Not Email', 'Lead Origin_Lead Add Form',
    'Lead Source_Welingak Website', 'Last Activity_Email Bounced',
    'Last Activity_Olark Chat Conversation',
    'Last Notable Activity_Email Link Clicked',
    'Last Notable Activity_Email Opened',
    'Last Notable Activity_Had a Phone Conversation',
    'Last Notable Activity_Modified',
    'Last Notable Activity_Olark Chat Conversation',
    'Last Notable Activity_Page Visited on Website'
]

st.title("Lead Conversion Prediction App")
st.write("This app predicts the probability of lead conversion based on the provided features.")

# User input features
lead_origin = st.selectbox('Lead Origin', ['Lead Add Form', 'Other'])
lead_source = st.selectbox('Lead Source', ['Welingak Website', 'Google', 'Facebook', 'Referral', 'Others'])
last_activity = st.selectbox('Last Activity', ['Email Bounced', 'Olark Chat Conversation', 'Email Opened', 'SMS Sent'])
do_not_email = st.radio('Do Not Email', ['Yes', 'No'])
last_notable_activity = st.selectbox('Last Notable Activity', [
    'Email Link Clicked', 'Email Opened', 'Had a Phone Conversation', 'Modified', 
    'Olark Chat Conversation', 'Page Visited on Website'])

# Create a dictionary of user input with binary and one-hot encoding
user_data = {
    'Do Not Email': 1 if do_not_email == 'Yes' else 0,
    'Lead Origin_Lead Add Form': 1 if lead_origin == 'Lead Add Form' else 0,
    'Lead Source_Welingak Website': 1 if lead_source == 'Welingak Website' else 0,
    'Last Activity_Email Bounced': 1 if last_activity == 'Email Bounced' else 0,
    'Last Activity_Olark Chat Conversation': 1 if last_activity == 'Olark Chat Conversation' else 0,
    'Last Notable Activity_Email Link Clicked': 1 if last_notable_activity == 'Email Link Clicked' else 0,
    'Last Notable Activity_Email Opened': 1 if last_notable_activity == 'Email Opened' else 0,
    'Last Notable Activity_Had a Phone Conversation': 1 if last_notable_activity == 'Had a Phone Conversation' else 0,
    'Last Notable Activity_Modified': 1 if last_notable_activity == 'Modified' else 0,
    'Last Notable Activity_Olark Chat Conversation': 1 if last_notable_activity == 'Olark Chat Conversation' else 0,
    'Last Notable Activity_Page Visited on Website': 1 if last_notable_activity == 'Page Visited on Website' else 0
}

input_df = pd.DataFrame([user_data])

# Align the input data with the selected features expected by the model
input_df = input_df.reindex(columns=selected_features, fill_value=0)

# Preprocess the input data (apply imputation)
input_df_cleaned = pd.DataFrame(mode_imputer.transform(input_df), columns=selected_features)

# Make predictions
if st.button("Predict"):
    prediction_prob = model.predict_proba(input_df_cleaned)[:, 1][0]
    st.subheader("Prediction Result")
    st.write(f"The predicted probability of lead conversion is **{prediction_prob*100:.2f}%**.")
