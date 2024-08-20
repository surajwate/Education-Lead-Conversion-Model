from src.logging_utils import configure_logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import logging
from sklearn.feature_selection import RFE

# Configure the logger
logger = configure_logging(log_file_name="model_building.log")

# Set logging level for this specific module
# logger.setLevel(logging.DEBUG)  # Uncomment this to enable DEBUG just for this module

def train_logistic_regression_rfe(X_train, y_train, n_features_to_select=11, fold=100, save=True):
    logger.info(f"Fold {fold}: Starting logistic regression training with RFE for {n_features_to_select} features...")

    # Initialize the model and RFE
    model = LogisticRegression()
    rfe = RFE(model, n_features_to_select=n_features_to_select)
    
    # Fit RFE
    rfe.fit(X_train, y_train)
    logger.info(f"Fold {fold}: RFE fitting completed. Feature ranking obtained.")
    
    # Extract the ranking of features
    ranking = pd.Series(rfe.ranking_, index=X_train.columns)
    logger.debug(f"Fold {fold}: Feature ranking:\n{ranking.sort_values()}")

    # Train model on selected features
    selected_features = X_train.columns[rfe.support_]
    model.fit(X_train[selected_features], y_train)
    logger.info(f"Fold {fold}: Model training completed using selected features: \n{selected_features}")
    
    if save:
        # Save the model
        model_path = f"./models/log_reg_rfe_fold_{fold}.pkl"
        try:
            joblib.dump(model, model_path)
            logger.info(f"Fold {fold}: Model successfully saved to {model_path}")
        except Exception as e:
            logger.error(f"Fold {fold}: Failed to save the model to {model_path}. Error: {e}")
            raise

        # Save the selected features
        feature_path = f"./models/log_reg_rfe_features_fold_{fold}.pkl"
        joblib.dump(selected_features.tolist(), feature_path)
        logger.info(f"Fold {fold}: Selected features saved to {feature_path}")

        # Save the feature ranking
        ranking_path = f"./models/log_reg_rfe_ranking_fold_{fold}.pkl"
        joblib.dump(ranking, ranking_path)
        logger.info(f"Fold {fold}: Feature ranking saved to {ranking_path}")
    
    return {
        "model": model,
        "selected_features": selected_features.tolist()
    }

def load_trained_model_rfe(fold):
    # Load the saved model
    model_path = f"./models/log_reg_rfe_fold_{fold}.pkl"
    model = joblib.load(model_path)

    # Load the saved feature names
    feature_path = f"./models/log_reg_rfe_features_fold_{fold}.pkl"
    selected_features = joblib.load(feature_path)

    # Load the feature ranking
    ranking_path = f"./models/log_reg_rfe_ranking_fold_{fold}.pkl"
    ranking = joblib.load(ranking_path)

    return model, selected_features, ranking