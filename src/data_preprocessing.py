import pandas as pd
import logging
from src.logging_utils import configure_logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = configure_logging(log_file_name="data_preprocessing.log")

def load_data(input_path):
    logger.info(f"Loading cleaned data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    return df

def preprocess_data(df, fold):
    logger.info(f"Starting preprocessing of data for fold {fold}...")

    # Mapping binary columns to 0/1
    binary_mapping = {'Yes': 1, 'No': 0}
    binary_columns = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype == 'O']
    if 'Converted' in binary_columns:
        binary_columns.remove('Converted')
    df[binary_columns] = df[binary_columns].apply(lambda x: x.map(binary_mapping))
    logger.info(f"Binary columns mapped: {binary_columns}. Data shape: {df.shape}")

    # Converting specific object columns to numeric
    numerical_columns = ['Converted', 'TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit', 'kfold']
    df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')
    logger.info(f"Numerical columns converted: {numerical_columns}. Data shape: {df.shape}")

    # Removing 'Converted' and 'kfold' columns from numerical columns to be scaled
    numerical_columns.remove('Converted')
    numerical_columns.remove('kfold')

    # Split data into train and test based on kfold column
    train_data = df[df.kfold != fold].reset_index(drop=True)
    test_data = df[df.kfold == fold].reset_index(drop=True)
    logger.info(f"Split data into train and test. Train shape: {train_data.shape}, Test shape: {test_data.shape}")

    # Scale the numerical columns
    scaler = StandardScaler()
    train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])
    test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])
    logger.info("Numerical columns scaled.")

    # One-hot encode categorical columns
    categorical_columns = [col for col in df.columns if df[col].dtype == 'O']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    
    # Encode and merge categorical columns
    encoded_train = pd.DataFrame(encoder.fit_transform(train_data[categorical_columns]), columns=encoder.get_feature_names_out(categorical_columns))
    encoded_test = pd.DataFrame(encoder.transform(test_data[categorical_columns]), columns=encoder.get_feature_names_out(categorical_columns))

    train_data = train_data.drop(categorical_columns, axis=1)
    test_data = test_data.drop(categorical_columns, axis=1)
    
    train_data = pd.concat([train_data, encoded_train], axis=1)
    test_data = pd.concat([test_data, encoded_test], axis=1)
    logger.info(f"Categorical columns encoded. Train shape: {train_data.shape}, Test shape: {test_data.shape}")

    # Concatenate the final train and test data
    final_df = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
    logger.info(f"Preprocessing completed for fold {fold}. Final data shape: {final_df.shape}")

    return final_df

if __name__ == "__main__":
    from data_cleaning import drop_columns, impute_missing_values
    configure_logging()
    logger.info("Starting data_preprocessing.py script.")
    
    try:
        data = pd.read_csv("./input/train_folds.csv")
        logger.info(f"Data loaded. Initial shape: {data.shape}")
        
        df = drop_columns(data)
        logger.info("Dropped unnecessary columns.")
        
        first_fold = impute_missing_values(df, 0)
        logger.info(f"Missing values in first fold: {first_fold.isna().sum().sum()} before preprocessing")
        
        # Example of processing a single fold (e.g., fold 0)
        preprocessed_df = preprocess_data(first_fold, fold=0)
        logger.info(f"Missing values in first fold after preprocessing: {preprocessed_df.isna().sum().sum()}")
        
        logger.info("Data preprocessing pipeline executed successfully.")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
    
    logger.info("data_preprocessing.py script completed successfully.")
