import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import logging
from src.logging_utils import configure_logging
from pathlib import Path
from typing import Union

# Configure the logger for this specific module
logger = configure_logging(log_file_name="create_folds.log")

def create_folds(
    input_path: Path,
    output_path: Path,
    n_splits: int = 5,
    stratify: bool = False,
    target_column: str = None,
    random_state: int = 42,
) -> None:
    """
    Create k-fold splits (stratified or not) and save the result.

    Args:
        input_path (Path): Path to the input CSV file.
        output_path (Path): Path to save the output CSV file.
        n_splits (int): Number of splits for k-fold. Defaults to 5.
        stratify (bool): Whether to use stratified k-fold. Defaults to False.
        target_column (str): Name of the target column for stratification.
                             Required if stratify is True.
        random_state (int): Random state for reproducibility. Defaults to 42.
    """
    try:
        logger.info(f"Reading data from {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")

        # Create a kfold column
        df["kfold"] = -1
        logger.info("Initialized kfold column.")

        # Randomize the rows
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        logger.info("Data randomized.")

        # Choose between StratifiedKFold and KFold
        if stratify:
            if target_column is None or target_column not in df.columns:
                error_message = f"Target column '{target_column}' not found or not specified for stratified split."
                logger.error(error_message)
                raise ValueError(error_message)

            y = df[target_column].values
            splitter: Union[KFold, StratifiedKFold] = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            logger.info("StratifiedKFold will be used for splitting.")
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            logger.info("KFold will be used for splitting.")

        # Fill the new kfold column
        for fold, (_, val_idx) in enumerate(splitter.split(X=df, y=(y if stratify else None))):
            df.loc[val_idx, "kfold"] = fold
            logger.info(f"Assigned fold {fold} to {len(val_idx)} samples.")

        # Save the dataframe
        df.to_csv(output_path, index=False)
        logger.info(f"Folded data saved to {output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
