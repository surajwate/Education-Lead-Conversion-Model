from src.logging_utils import configure_logging
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.feature_selection import RFE

logger = configure_logging(log_file_name="model_building.log")

def train_logistic_regression_rfe(X_train, y_train, n_features_to_select=11, fold=0):
    logger.info("Starting logistic regression training with RFE...")
    model = LogisticRegression()
    rfe = RFE(model, n_features_to_select=n_features_to_select)
    rfe.fit(X_train, y_train)
    logger.info("RFE training completed.")
    
    # Extract the ranking of features
    ranking = pd.Series(rfe.ranking_, index=X_train.columns)
    logger.info(f"Feature ranking:\n{ranking.sort_values()}")

    # Continue with the model trained on selected features
    selected_features = X_train.columns[rfe.support_]
    model.fit(X_train[selected_features], y_train)
    
    # Save the model with a specific name including the fold number
    model_path = f"./models/log_reg_rfe_fold_{fold}.pkl"
    logger.info(f"Attempting to save model to: {model_path}")
    joblib.dump(model, model_path)
    logger.info(f"Model successfully saved to {model_path}")

    return model, selected_features

