import pandas as pd
import joblib
from src.logging_utils import configure_logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Configure the logger for this specific module
logger = configure_logging(log_file_name="data_preprocessing.log")

def load_data(input_path):
    logger.info(f"Loading cleaned data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    return df

def preprocess_data(df, fold, save=True):
    logger.info(f"Starting preprocessing of data for fold {fold}...")

    # Define and save binary mapping configuration
    binary_mapping = {'Yes': 1, 'No': 0}
    binary_columns = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype == 'O']
    if 'Converted' in binary_columns:
        binary_columns.remove('Converted')

    if save:
        joblib.dump(binary_mapping, f"./models/binary_mapping_fold_{fold}.pkl")
        joblib.dump(binary_columns, f"./models/binary_columns_fold_{fold}.pkl")

    # Apply binary mapping to binary columns
    df[binary_columns] = df[binary_columns].apply(lambda x: x.map(binary_mapping))
    logger.info(f"Binary columns mapped: {binary_columns}. Data shape: {df.shape}")

    # Save and convert numerical columns
    numerical_columns = ['Converted', 'TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit', 'kfold']
    if save:
        joblib.dump(numerical_columns, f"./models/numerical_columns_fold_{fold}.pkl")

    df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')
    logger.info(f"Numerical columns converted: {numerical_columns}. Data shape: {df.shape}")

    # Removing 'Converted' and 'kfold' columns from numerical columns to be scaled
    numerical_columns.remove('Converted')
    numerical_columns.remove('kfold')

    # Split data into train and test based on kfold column
    train_data = df[df.kfold != fold].reset_index(drop=True)
    test_data = df[df.kfold == fold].reset_index(drop=True)
    logger.info(f"Split data into train and test. Train shape: {train_data.shape}, Test shape: {test_data.shape}")

    # One-hot encode categorical columns and initialize the encoder
    categorical_columns = [col for col in df.columns if df[col].dtype == 'O']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')

    # Fit OHE on training + test data to ensure no unseen categories in test
    full_data = pd.concat([train_data[categorical_columns], test_data[categorical_columns]], axis=0)
    encoder.fit(full_data)

    if save:
        # Save the encoder
        encoder_path = f"./models/encoder_fold_{fold}.pkl"
        joblib.dump(encoder, encoder_path)
        logger.info(f"Encoder saved to {encoder_path}")

    # Transform train and test data
    train_data_encoded = pd.DataFrame(encoder.transform(train_data[categorical_columns]).astype(int), 
                                      columns=encoder.get_feature_names_out(categorical_columns))
    test_data_encoded = pd.DataFrame(encoder.transform(test_data[categorical_columns]).astype(int), 
                                     columns=encoder.get_feature_names_out(categorical_columns))

    train_data = pd.concat([train_data, train_data_encoded], axis=1)
    test_data = pd.concat([test_data, test_data_encoded], axis=1)

    # Drop original categorical columns after encoding
    train_data = train_data.drop(categorical_columns, axis=1)
    test_data = test_data.drop(categorical_columns, axis=1)

    logger.info(f"Categorical columns encoded: {categorical_columns}.")

    # Scale the numerical columns
    scaler = StandardScaler()
    train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])
    test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])

    if save:
        # Save the scaler
        scaler_path = f"./models/scaler_fold_{fold}.pkl"
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

    logger.info("Numerical columns scaled.")

    # Debugging: Check if the target variable remains binary after preprocessing
    logger.info(f"Unique values in 'Converted' after preprocessing: {df['Converted'].unique()}")

    # Concatenate the final train and test data (ensure order is preserved)
    final_df = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
    logger.info(f"Preprocessing completed for fold {fold}. Final data shape: {final_df.shape}")

    return final_df

def load_preprocessing_step(fold):
    # Load all necessary preprocessing objects
    binary_mapping = joblib.load(f"./models/binary_mapping_fold_{fold}.pkl")
    binary_columns = joblib.load(f"./models/binary_columns_fold_{fold}.pkl")
    numerical_columns = joblib.load(f"./models/numerical_columns_fold_{fold}.pkl")
    encoder = joblib.load(f"./models/encoder_fold_{fold}.pkl")
    scaler = joblib.load(f"./models/scaler_fold_{fold}.pkl")
    return binary_mapping, binary_columns, numerical_columns, encoder, scaler