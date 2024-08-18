from src.logging_utils import configure_logging
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import joblib

configure_logging(log_file_name="model_building.log")


def train_logistic_regression(X_train, y_train, fold=0):
    logging.info("Starting logistic regression training...")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    logging.info("Logistic regression training completed.")
    
    # Save the model with a specific name including the fold number
    model_path = f"./models/log_reg_fold_{fold}.pkl"
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}.")
    
    return model

