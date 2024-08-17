import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib  # For saving/loading the model

# Configure logging
def configure_logging(log_file_name="model_building.log"):
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / log_file_name,
        level=logging.INFO,  # Change to DEBUG for more detailed logs
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",  # Use 'w' to overwrite the log file on each run
    )
    logging.info("Logging is configured.")

def load_data(input_path):
    logging.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    logging.info(f"Data loaded successfully. Shape: {df.shape}")
    return df

def train_model(X_train, y_train):
    logging.info("Starting model training...")
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    logging.info("Model training completed.")
    return log_reg

def evaluate_model(model, X_test, y_test):
    logging.info("Starting model evaluation...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    logging.info(f"Model Evaluation Completed.")
    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"Classification Report:\n{classification_rep}")
    logging.info(f"Confusion Matrix:\n{confusion_mat}")
    
    return accuracy, classification_rep, confusion_mat

if __name__ == "__main__":
    from data_cleaning import drop_columns, impute_missing_values
    from data_preprocessing import preprocess_data
    configure_logging()

    logging.info("Starting log_reg_model.py script.")
    
    try:
        # Load the data
        data = load_data("./input/train_folds.csv")
        
        # Clean and preprocess the data
        df = drop_columns(data)
        logging.info("Dropped unnecessary columns.")
        
        # Loop through each fold for cross-validation
        for fold in range(5):
            logging.info(f"Processing fold {fold}...")
            
            # Prepare the data for the current fold
            fold_df = impute_missing_values(df, fold)
            fold_df = preprocess_data(fold_df, fold)
            
            # Drop kfold column and split the data
            X = fold_df.drop(["Converted", "kfold"], axis=1)
            y = fold_df["Converted"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model
            model = train_model(X_train, y_train)
            
            # Save the model (optional)
            model_path = f"./models/log_reg_fold_{fold}.pkl"
            joblib.dump(model, model_path)
            logging.info(f"Model for fold {fold} saved to {model_path}.")
            
            # Evaluate the model
            accuracy, report, matrix = evaluate_model(model, X_test, y_test)
            
            # Print the results to the console
            print(f"Fold {fold} Results:")
            print(f"Accuracy: {accuracy}")
            print(f"Classification Report:\n{report}")
            print(f"Confusion Matrix:\n{matrix}")
            
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise
    
    logging.info("log_reg_model.py script completed successfully.")
