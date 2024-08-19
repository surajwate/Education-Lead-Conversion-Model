from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.log_reg_model import train_logistic_regression
from src.log_reg_rfe_model import train_logistic_regression_rfe

# Dictionary to store models and associated training functions
models = {
    "log_reg": {
        "model": LogisticRegression,
        "train_func": train_logistic_regression,
    },
    "log_reg_rfe": {
        "model": LogisticRegression,  # This could be another model if you use a different one for RFE
        "train_func": train_logistic_regression_rfe,
        "params": {"n_features_to_select": 11}  # Example parameter for RFE
    },
    # You can add more models here
    "random_forest": {
        "model": RandomForestClassifier,
        "train_func": None  # Assuming you have a separate function for training RandomForest
    }
}
