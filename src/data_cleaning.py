import numpy as np
import pandas as pd
import joblib
from src.logging_utils import configure_logging
from sklearn.impute import SimpleImputer

# Configure the logger for this specific module
logger = configure_logging(log_file_name="data_cleaning.log")

def drop_columns(df, save=True):
    logger.info("Dropping irrelevant columns...")
    initial_shape = df.shape

    # Only drop columns if they exist in the DataFrame
    columns_to_drop = ['Prospect ID', 'Lead Number']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]

    if save:
        # Save the columns to drop
        joblib.dump(columns_to_drop, "./models/columns_to_drop.pkl")
    
    # Drop the irrelevant columns
    if columns_to_drop:
        df.drop(columns=columns_to_drop, axis=1, inplace=True)
        logger.info(f"Dropped irrelevant columns: {columns_to_drop}. Shape before: {initial_shape}, after: {df.shape}")
    else:
        logger.info("No columns to drop. Shape remains unchanged.")

    # Handle "Select" values as nulls
    df.replace("Select", np.nan, inplace=True)  
    logger.info("Replaced 'Select' values with NaN.")

    # Drop the columns with more than 25% missing values
    threshold = 0.75
    df.dropna(thresh=threshold * df.shape[0], axis=1, inplace=True)
    logger.info(f"Dropped columns with more than {(1-threshold)*100}% missing values. Current shape: {df.shape}")

    # Drop the columns with a single unique value
    single_value_columns = [col for col in df.columns if df[col].dropna().nunique() == 1]
    df.drop(columns=single_value_columns, axis=1, inplace=True)
    logger.info(f"Dropped columns with single unique values: {single_value_columns}. Current shape: {df.shape}")

    return df

def impute_missing_values(df, fold=None, save=True):
    logger.info(f"Starting data imputation for fold {fold}...")
    
    # Create a copy of the dataframe to avoid any changes in the original dataframe
    df = df.copy()

    # If the kfold column is present, split the data based on the fold number; otherwise, use the entire dataset
    if "kfold" in df.columns:
        test = df[df.kfold == fold]
        train = df[df.kfold != fold]
        logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
    else:
        train = df
        test = pd.DataFrame() # Empty DataFrame for test data
        logger.info(f"Full dataset training. Train shape: {train.shape}")

    # Imputing the missing values with most frequent values
    mode_imputer = SimpleImputer(strategy='most_frequent')
    train = pd.DataFrame(mode_imputer.fit_transform(train), columns=df.columns)

    if not test.empty:
        test = pd.DataFrame(mode_imputer.transform(test), columns=df.columns)
        # Concatenate train and test DataFrames vertically
        combined_df = pd.concat([train, test], axis=0).reset_index(drop=True)
        logger.info(f"Data imputation completed for fold {fold}. Combined shape: {combined_df.shape}")
    else:
        combined_df = train.reset_index(drop=True)

    logger.info("Imputed missing values with most frequent values.")

    if save:
        # Save the fitted imputer
        imputer_path = f"./models/imputer_fold_{fold}.pkl"
        joblib.dump(mode_imputer, imputer_path)

    
    

    return combined_df

def load_dropped_columns_and_imputer(fold):
    # Load the columns to drop
    columns_to_drop = joblib.load("./models/columns_to_drop.pkl")

    # Load the imputer for the specific fold
    imputer_path = f"./models/imputer_fold_{fold}.pkl"
    mode_imputer = joblib.load(imputer_path)

    return columns_to_drop, mode_imputer