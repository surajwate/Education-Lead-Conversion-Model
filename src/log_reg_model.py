import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configure logging
def configure_logging():
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "model_building.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",
    )

def load_data(input_path):
    logging.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    logging.info(f"Data loaded with shape: {df.shape}")
    return df

def train_and_evaluate_model(df, fold):
    logging.info(f"Starting model training and evaluation for fold {fold}...")
    
    # Impute missing values
    fold_df = impute_missing_values(df, fold)
    
    # Preprocess the data
    fold_df = preprocess_data(fold_df, fold)
    
    # Drop the kfold column
    fold_df = fold_df.drop("kfold", axis=1)
    
    # Split the data into X and y
    X = fold_df.drop("Converted", axis=1)
    y = fold_df["Converted"]
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the logistic regression model
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    
    # Make predictions
    y_pred = log_reg.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy for fold {fold}: {accuracy}")
    logging.info(f"Classification Report for fold {fold}:\n{classification_report(y_test, y_pred)}")
    logging.info(f"Confusion Matrix for fold {fold}:\n{confusion_matrix(y_test, y_pred)}")

    return accuracy, classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    from data_cleaning import drop_columns, impute_missing_values
    from data_preprocessing import preprocess_data
    configure_logging()

    data = load_data("./input/train_folds.csv")
    df = drop_columns(data)

    # Loop over each fold
    for fold in range(5):
        accuracy, report, matrix = train_and_evaluate_model(df, fold)
        print(f"Fold {fold} Results:")
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")
        print(f"Confusion Matrix:\n{matrix}")
