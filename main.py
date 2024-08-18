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

configure_logging(log_file_name="main_pipeline.log")


def process_fold(fold, df):
    logging.info(f"Processing fold {fold}...")
    
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
    accuracy, report, matrix = evaluate_model(model, X_test, y_test)
    
    # Return accuracy and other evaluation results
    return accuracy, report, matrix

def main():
    configure_logging()
    start_time = time.time()
    logging.info("Starting main pipeline.")
    
    try:
        # Step 1: Create K-Folds
        input_path = Path("./input/Leads.csv")
        output_path = Path("./input/train_folds.csv")
        create_folds(input_path, output_path, stratify=True, target_column="Converted")
        
        # Step 2: Data Cleaning
        data = load_data(output_path)
        df = drop_columns(data)

        all_accuracies = []
        
        for fold in range(5):
            accuracy, report, matrix = process_fold(fold, df)
            all_accuracies.append(accuracy)
            
            # Print results for each fold
            print(f"Fold {fold} Results:")
            print(f"Accuracy: {accuracy}")
            print(f"Classification Report:\n{report}")
            print(f"Confusion Matrix:\n{matrix}")

        logging.info(f"Average Accuracy: {sum(all_accuracies) / len(all_accuracies):.2f}")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise
    
    logging.info("Main pipeline completed successfully.")
    end_time = time.time()
    logging.info(f"Execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
