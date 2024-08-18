from src.logging_utils import configure_logging
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import joblib

logger = configure_logging(log_file_name="model_building.log")


def train_logistic_regression(X_train, y_train, fold=0):
    logger.info("Starting logistic regression training...")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    logger.info("Logistic regression training completed.")
    
    # Save the model with a specific name including the fold number
    model_path = f"./models/log_reg_fold_{fold}.pkl"
    logger.info(f"Attempting to save model to: {model_path}")
    joblib.dump(model, model_path)
    logger.info(f"Model successfully saved to {model_path}")

    logger.info(f"Model saved to {model_path}.")
    
    return model

