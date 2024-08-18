# model_utils.py
import logging
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def configure_logging(log_file_name="pipeline.log"):
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / log_file_name,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",
    )
    logging.info("Logging is configured.")

def load_data(input_path):
    logging.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    logging.info(f"Data loaded successfully. Shape: {df.shape}")
    return df

def evaluate_model(model, X_test, y_test):
    logging.info("Starting model evaluation...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    logging.info("Model Evaluation Completed.")
    logging.info("Accuracy: %s", accuracy)
    logging.info("Classification Report:\n%s", classification_rep) 
    logging.info("Confusion Matrix:\n%s", confusion_mat)            
    
    return accuracy, classification_rep, confusion_mat

