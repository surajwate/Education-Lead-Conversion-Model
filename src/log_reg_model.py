from src.logging_utils import configure_logging
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import joblib

logger = configure_logging(log_file_name="model_building.log")


def train_logistic_regression(X_train, y_train, fold=0, save=True):
    logger.info("Starting logistic regression training...")

    # Instantiate and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    logger.info("Logistic regression training completed.")
    
    if save:
        # Save the model with a specific name including the fold number
        model_path = f"./models/log_reg_fold_{fold}.pkl"
        logger.info(f"Attempting to save model to: {model_path}")
        joblib.dump(model, model_path)
        logger.info(f"Model successfully saved to {model_path}")

        # Save the feature names used for training
        feature_path = f"./models/log_reg_features_fold_{fold}.pkl"
        joblib.dump(X_train.columns.tolist(), feature_path)
        logger.info(f"Feature names saved to {feature_path}")
    
    return {
        "model": model,
        "selected_features": X_train.columns.tolist()    # All features are selected
    }

def load_trained_model(fold):
    # Load the saved model
    model_path = f"./models/log_reg_fold_{fold}.pkl"
    model = joblib.load(model_path)

    # Load the saved feature names
    feature_path = f"./models/log_reg_features_fold_{fold}.pkl"
    selected_features = joblib.load(feature_path)

    return model, selected_features