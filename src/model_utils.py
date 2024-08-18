import logging
from src.logging_utils import configure_logging
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

logger = configure_logging(log_file_name="evaluation.log")

def load_data(input_path):
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    return df

def evaluate_model(model, X_test, y_test):
    logger.info("Starting model evaluation...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    logger.info("Model Evaluation Completed.")
    logger.info("Accuracy: %s", accuracy)
    logger.info("Classification Report:\n%s", classification_rep) 
    logger.info("Confusion Matrix:\n%s", confusion_mat)            
    
    return accuracy, classification_rep, confusion_mat

