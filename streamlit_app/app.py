import streamlit as st
import joblib
import pandas as pd

# Load the trained model and preprocessing objects
model = joblib.load("./models/log_reg_rfe_fold_100.pkl")
scaler = joblib.load("./models/scaler_fold_100.pkl")
encoder = joblib.load("./models/encoder_fold_100.pkl")
binary_mapping = joblib.load("./models/binary_mapping_fold_100.pkl")
binary_columns = joblib.load("./models/binary_columns_fold_100.pkl")
numerical_columns = joblib.load("./models/numerical_columns_fold_100.pkl")
selected_features = joblib.load("./models/log_reg_rfe_features_fold_100.pkl")

# Remove the 'Converted' column from numerical_columns if it's present
if 'Converted' in numerical_columns:
    numerical_columns.remove('Converted')

# Streamlit app title
st.title("Lead Scoring Prediction App")

# Collect user inputs for each selected feature
do_not_email = st.selectbox("Do Not Email", ("Yes", "No"))
total_time_spent = st.number_input("Total Time Spent on Website", min_value=0.0, max_value=1000.0, value=100.0)

# Collect inputs for categorical features
lead_origin = st.selectbox("Lead Origin", ["Lead Add Form", "Landing Page Submission", "API", "Lead Import"])
lead_source = st.selectbox("Lead Source", ["Olark Chat", "Organic Search", "Direct Traffic", "Google", "Welingak Website", "Referral Sites"])
last_activity = st.selectbox("Last Activity", ["Email Opened", "Page Visited on Website", "Olark Chat Conversation", "Converted to Lead"])
last_notable_activity = st.selectbox("Last Notable Activity", ["Email Link Clicked", "Email Opened", "Modified", "Olark Chat Conversation", "Page Visited on Website"])

# Manually create the input data
input_data = {
    "Do Not Email": binary_mapping[do_not_email],
    "TotalVisits": 0,  # Default to 0, as it might not be used
    "Page Views Per Visit": 0,  # Default to 0, as it might not be used
    "Total Time Spent on Website": total_time_spent,
    "Lead Origin_Lead Add Form": 1 if lead_origin == "Lead Add Form" else 0,
    "Lead Source_Olark Chat": 1 if lead_source == "Olark Chat" else 0,
    "Lead Source_Welingak Website": 1 if lead_source == "Welingak Website" else 0,
    "Last Activity_Olark Chat Conversation": 1 if last_activity == "Olark Chat Conversation" else 0,
    "Last Notable Activity_Email Link Clicked": 1 if last_notable_activity == "Email Link Clicked" else 0,
    "Last Notable Activity_Email Opened": 1 if last_notable_activity == "Email Opened" else 0,
    "Last Notable Activity_Modified": 1 if last_notable_activity == "Modified" else 0,
    "Last Notable Activity_Olark Chat Conversation": 1 if last_notable_activity == "Olark Chat Conversation" else 0,
    "Last Notable Activity_Page Visited on Website": 1 if last_notable_activity == "Page Visited on Website" else 0,
}

# Convert the input_data dictionary to a DataFrame
user_data = pd.DataFrame([input_data])

# Debugging: Print the columns of user_data and selected_features to ensure they match
# st.write("User Data Columns: ", user_data.columns.tolist())
# st.write("Selected Features: ", selected_features)

# Reorder the columns to match the order expected by the model
user_data = user_data[selected_features + [col for col in user_data.columns if col not in selected_features]]

# Ensure the columns are in the correct order for the scaler and model
user_data[numerical_columns] = scaler.transform(user_data[numerical_columns])

# Apply the model to the user data
if st.button("Predict"):
    prediction = model.predict(user_data[selected_features])
    prediction_proba = model.predict_proba(user_data[selected_features])[:, 1]

    # Display the result
    if prediction[0] == 1:
        st.success(f"The model predicts that the lead is likely to convert with a probability of {prediction_proba[0]:.2f}.")
    else:
        st.warning(f"The model predicts that the lead is unlikely to convert with a probability of {1 - prediction_proba[0]:.2f}.")
