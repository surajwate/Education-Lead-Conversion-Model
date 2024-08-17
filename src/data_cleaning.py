import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.impute import SimpleImputer

# Configure logging
def configure_logging():
    # Ensure the logs directory exists
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging to file in the logs directory
    log_file = "logs/data_cleaning.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",  # Use 'w' to overwrite the log file on each run
    )

def drop_columns(df):

    # drop the irrelevant columns
    df.drop(['Prospect ID', 'Lead Number'], axis=1, inplace=True)
    logging.info("Irrelevant columns dropped.")

    # As "Select" Value of many of columns are as good as null, we will replace "Select" values with null
    df.replace("Select", np.nan, inplace=True)  

    # drop the columsn with more than 25% missing values
    threshold = 0.75
    df.dropna(thresh=threshold*df.shape[0], axis=1, inplace=True)
    logging.info(f"Dropped columns with more than {(1-threshold)*100}% missing values.")

    # drop the columns with single value
    single_value_columns = [col for col in df.columns if df[col].dropna().nunique() == 1]
    df.drop(columns=single_value_columns, axis=1, inplace=True)
    logging.info(f"Dropped columns with single unique values: {single_value_columns}")

    return df

# Impute missing values for each fold
def impute_missing_values(df, fold):
    logging.info(f"Starting data cleaning for fold {fold}...")

    # Create a copy of the dataframe to avoid any changes in the original dataframe
    df = df.copy()

    test = df[df.kfold == fold]
    train = df[df.kfold != fold]
    logging.info(f"Train shape: {train.shape}, Test shape: {test.shape}")

    # Imputing the missing values with most frequent values
    mode_imputer = SimpleImputer(strategy='most_frequent')
    train = mode_imputer.fit_transform(train)
    test = mode_imputer.transform(test)
    logging.info("Imputed missing values with most frequent values.")

    # Converting the numpy array back to pandas dataframe
    train = pd.DataFrame(train, columns=df.columns)
    test = pd.DataFrame(test, columns=df.columns)

    # Concatenate train and test DataFrames vertically
    combined_df = pd.concat([train, test], axis=0).reset_index(drop=True)

    return combined_df

if __name__ == "__main__":
    configure_logging()
    
    # Load the data
    data = pd.read_csv("./input/train_folds.csv")
    data = drop_columns(data)

    for fold in range(5):
        fold_df = impute_missing_values(data, fold)
        logging.info(f"Data cleaning completed for fold {fold}.")