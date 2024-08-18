import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.logging_utils import configure_logging
from src.create_folds import create_folds
from src.data_cleaning import drop_columns, impute_missing_values
from src.data_preprocessing import preprocess_data
from src.log_reg_model import train_logistic_regression
from src.model_utils import evaluate_model, load_data

import time

logger = configure_logging(log_file_name="main_pipeline.log")


def process_fold(fold, df):
    logger.info(f"Processing fold {fold}...")
    
    # Step 3: Data Imputation
    fold_df = impute_missing_values(df, fold)
    
    # Step 4: Data Preprocessing
    fold_df = preprocess_data(fold_df, fold)
    
    # Step 5: Model Training
    X = fold_df.drop(["Converted", "kfold"], axis=1)
    y = fold_df["Converted"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_logistic_regression(X_train, y_train, fold=fold)

    
    # Step 6: Model Evaluation
    accuracy, precision, roc_auc = evaluate_model(model, X_test, y_test, threshold=0.62)
    
    # Return accuracy and other evaluation results
    return accuracy, precision, roc_auc

def main():
    configure_logging()
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
            accuracy, precision, roc_auc = process_fold(fold, df)
            all_accuracies.append(accuracy)
            all_precisions.append(precision)
            
            # Print results for each fold
            print(f"Fold {fold} Results: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, ROC AUC: {roc_auc:.4f}")
            # print(f"Classification Report:\n{report}")
            # print(f"Confusion Matrix:\n{matrix}")

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
