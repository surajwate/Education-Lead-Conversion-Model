import logging
import time
from pathlib import Path
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from src.logging_utils import configure_logging
from src.create_folds import create_folds
from src.data_cleaning import drop_columns, impute_missing_values
from src.data_preprocessing import preprocess_data
from src.log_reg_model import train_logistic_regression
from src.log_reg_rfe_model import train_logistic_regression_rfe
from src.model_utils import evaluate_model, load_data
from src.model_dispatcher import models
import joblib

logger = configure_logging(log_file_name="main_pipeline.log")

def process_fold(fold, df, model_type):
    logger.info(f"Processing fold {fold} using {model_type}...")

    # Step 3: Data Imputation
    fold_df = impute_missing_values(df, fold)

    # Step 4: Data Preprocessing
    fold_df = preprocess_data(fold_df, fold)

    # Step 5: Split data based on the fold number
    train_df = fold_df[fold_df.kfold != fold].reset_index(drop=True)
    test_df = fold_df[fold_df.kfold == fold].reset_index(drop=True)

    X_train = train_df.drop(["Converted", "kfold"], axis=1)
    y_train = train_df["Converted"]
    X_test = test_df.drop(["Converted", "kfold"], axis=1)
    y_test = test_df["Converted"]

    # Retrieve the model and training function from the model_dispatcher
    model_info = models.get(model_type)

    if model_info is None:
        raise ValueError(f"Invalid model type: {model_type}. Choose from: {list(models.keys())}")

    train_func = model_info.get("train_func")

    if train_func:
        # Train the model using the custom training function
        result = train_func(X_train, y_train, fold=fold, **model_info.get("params", {}))
        model = result["model"]
        selected_features = result["selected_features"]
    else:
        # If there's no custom training function, instantiate and train the model directly
        model = model_info["model"]()
        model.fit(X_train, y_train)
        selected_features = X_train.columns.tolist()  # Use all features by default

    # Step 6: Model Evaluation
    accuracy, precision, roc_auc = evaluate_model(model, X_test[selected_features], y_test, threshold=0.62)

    # Return accuracy and other evaluation results
    return accuracy, precision, roc_auc

def train_on_full_data(df, model_type):
    logger.info(f"Training on full data using {model_type}...")

    # Step 3: Data Imputation
    df = impute_missing_values(df, fold=100)  # Dummy fold number for distinction

    # Step 4: Data Preprocessing
    df = preprocess_data(df, fold=100)  # Dummy fold number for distinction

    # Step 5: Split data into features and target
    X = df.drop(["Converted"], axis=1)
    y = df["Converted"]

    # Retrieve the model and training function from the model_dispatcher
    model_info = models.get(model_type)

    if model_info is None:
        raise ValueError(f"Invalid model type: {model_type}. Choose from: {list(models.keys())}")

    train_func = model_info.get("train_func")

    if train_func:
        # Train the model using the custom training function
        result = train_func(X, y, fold=100, **model_info.get("params", {}))
        model = result["model"]
        selected_features = result["selected_features"]
    else:
        # If there's no custom training function, instantiate and train the model directly
        model = model_info["model"]()
        model.fit(X, y)
        selected_features = X.columns.tolist()  # Use all features by default

    # Step 6: Model Evaluation
    accuracy, precision, roc_auc = evaluate_model(model, X[selected_features], y, threshold=0.62)

    # Save the final model and any preprocessing steps as needed
    model_path = f"./models/final_{model_type}_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Final model saved to {model_path}.")

    return accuracy, precision, roc_auc

def main():
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="validate", choices=["validate", "train_full"],
                        help="Specify the mode: 'validate' to run cross-validation, 'train_full' to train on the entire dataset.")
    parser.add_argument("--model_type", type=str, default="log_reg", choices=["log_reg", "log_reg_rfe"],
                        help="Specify the model type: 'log_reg' or 'log_reg_rfe'.")
    args = parser.parse_args()

    start_time = time.time()
    logger.info(f"Starting pipeline in {args.mode} mode.")

    try:
        input_path = Path("./input/Leads.csv")
        output_path = Path("./input/train_folds.csv")

        if args.mode == "validate":
            # Step 1: Create K-Folds
            create_folds(input_path, output_path, stratify=True, target_column="Converted")

            # Load the dataset with folds
            data = load_data(output_path)
            df = drop_columns(data)

            all_accuracies = []
            all_precisions = []

            for fold in range(5):
                accuracy, precision, roc_auc = process_fold(fold, df, model_type=args.model_type)
                all_accuracies.append(accuracy)
                all_precisions.append(precision)

                # Print results for each fold
                print(f"Fold {fold} Results: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, ROC AUC: {roc_auc:.4f}")

            logger.info(f"Average Accuracy: {sum(all_accuracies) / len(all_accuracies):.2f}")
            print(f"Average Accuracy: {sum(all_accuracies) / len(all_accuracies):.2f}")
            print(f"Average Precision: {sum(all_precisions) / len(all_precisions):.2f}")

        elif args.mode == "train_full":
            # Load the original dataset
            data = load_data(input_path)
            df = drop_columns(data)

            # Train on the full dataset
            accuracy, precision, roc_auc = train_on_full_data(df, model_type=args.model_type)

            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, ROC AUC: {roc_auc:.4f}")


    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

    logger.info(f"Pipeline completed successfully in {args.mode} mode.")
    end_time = time.time()
    logger.info(f"Execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
