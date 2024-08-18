import logging
from src.logging_utils import configure_logging
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score, 
    precision_recall_curve, f1_score, precision_score, recall_score
)

logger = configure_logging(log_file_name="evaluation.log")

def load_data(input_path):
    logger.info("="*50)
    logger.info("Loading Data")
    logger.info("="*50)
    
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Data loaded successfully with shape: {df.shape}")
    
    return df

def evaluate_model(model, X_test, y_test, threshold=0.5):
    logger.info("="*50)
    logger.info("Starting Model Evaluation")
    logger.info("="*50)
    
    logger.info(f"Using custom threshold: {threshold:.2f}")
    
    # Get probability scores for the positive class
    probs = model.predict_proba(X_test)[:, 1]
    
    # Apply the custom threshold
    y_pred = (probs >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, probs)
    classification_rep = classification_report(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    
    # Log the metrics in a structured format
    logger.info("Model Evaluation Metrics:")
    logger.info(f"  - Accuracy: {accuracy:.4f}")
    logger.info(f"  - Precision: {precision:.4f}")
    logger.info(f"  - Recall: {recall:.4f}")
    logger.info(f"  - F1 Score: {f1:.4f}")
    logger.info(f"  - ROC AUC: {roc_auc:.4f}")
    logger.info("  - Classification Report:\n" + classification_rep)
    logger.info("  - Confusion Matrix:\n" + "\n".join(
        [f"    {row}" for row in confusion_mat]))
    logger.info("="*50)
    
    logger.info("Model Evaluation Completed")
    
    return accuracy, precision, roc_auc
