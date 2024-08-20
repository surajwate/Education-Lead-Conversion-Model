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

logger = configure_logging(log_file_name="main_pipeline.log")


def process_fold(fold, df, model_type, n_features_to_select=None):
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
        selected_features = X_train.columns  # Use all features by default



    # if model_type == "log_reg":
    #     model = train_logistic_regression(X_train, y_train, fold=fold)
    #     selected_features = X_train.columns  # All features are selected
    # elif model_type == "log_reg_rfe":
    #     model, selected_features = train_logistic_regression_rfe(X_train, y_train, n_features_to_select=n_features_to_select, fold=fold)
    # else:
    #     raise ValueError("Invalid model type. Choose between 'log_reg' and 'log_reg_rfe'.")

    # Step 6: Model Evaluation
    accuracy, precision, roc_auc = evaluate_model(model, X_test[selected_features], y_test, threshold=0.62)

    # Return accuracy and other evaluation results
    return accuracy, precision, roc_auc


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, default="log_reg", choices=["log_reg", "log_reg_rfe"],
                        help="Specify the model type: 'log_reg' or 'log_reg_rfe'.")
    # parser.add_argument("--n_features", type=int, default=10, help="Number of features to select in RFE.")
    args = parser.parse_args()

    start_time = time.time()
    logger.info("Starting main pipeline.")

    try:
        # Step 1: Create K-Folds
        input_path = Path("./input/Leads.csv")
        output_path = Path("./input/train_folds.csv")
        create_folds(input_path, output_path, stratify=True, target_column="Converted")

        # Step 2: Data Cleaning
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

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

    logger.info("Main pipeline completed successfully.")
    end_time = time.time()
    logger.info(f"Execution time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
