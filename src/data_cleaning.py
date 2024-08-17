import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.impute import SimpleImputer

# Configure logging
def configure_logging(log_file_name="data_cleaning.log"):
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / log_file_name,
        level=logging.INFO,  # Use DEBUG if more detailed logs are required
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",  # Use 'w' to overwrite the log file on each run
    )
    logging.info("Logging is configured.")

def drop_columns(df):
    logging.info("Dropping irrelevant columns...")
    initial_shape = df.shape
    
    # drop the irrelevant columns
    df.drop(['Prospect ID', 'Lead Number'], axis=1, inplace=True)
    logging.info(f"Dropped irrelevant columns. Shape before: {initial_shape}, after: {df.shape}")

    # Handle "Select" values as nulls
    df.replace("Select", np.nan, inplace=True)  
    logging.info("Replaced 'Select' values with NaN.")

    # drop the columns with more than 25% missing values
    threshold = 0.75
    df.dropna(thresh=threshold * df.shape[0], axis=1, inplace=True)
    logging.info(f"Dropped columns with more than {(1-threshold)*100}% missing values. Current shape: {df.shape}")

    # drop the columns with single unique value
    single_value_columns = [col for col in df.columns if df[col].dropna().nunique() == 1]
    df.drop(columns=single_value_columns, axis=1, inplace=True)
    logging.info(f"Dropped columns with single unique values: {single_value_columns}. Current shape: {df.shape}")

    return df

def impute_missing_values(df, fold):
    logging.info(f"Starting data imputation for fold {fold}...")
    
    # Create a copy of the dataframe to avoid any changes in the original dataframe
    df = df.copy()

    # Split into train and test based on the fold
    test = df[df.kfold == fold]
    train = df[df.kfold != fold]
    logging.info(f"Train shape: {train.shape}, Test shape: {test.shape}")

    # Imputing the missing values with most frequent values
    mode_imputer = SimpleImputer(strategy='most_frequent')
    train = pd.DataFrame(mode_imputer.fit_transform(train), columns=df.columns)
    test = pd.DataFrame(mode_imputer.transform(test), columns=df.columns)
    logging.info("Imputed missing values with most frequent values.")

    # Concatenate train and test DataFrames vertically
    combined_df = pd.concat([train, test], axis=0).reset_index(drop=True)
    logging.info(f"Data imputation completed for fold {fold}. Combined shape: {combined_df.shape}")

    return combined_df

if __name__ == "__main__":
    configure_logging()
    logging.info("Starting data_cleaning.py script.")
    
    try:
        # Load the data
        data = pd.read_csv("./input/train_folds.csv")
        logging.info(f"Data loaded. Initial shape: {data.shape}")
        
        # Drop unnecessary columns
        data = drop_columns(data)
        
        # Impute missing values for each fold
        for fold in range(5):
            fold_df = impute_missing_values(data, fold)
            logging.info(f"Data cleaning completed for fold {fold}.")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise
    
    logging.info("data_cleaning.py script completed successfully.")
