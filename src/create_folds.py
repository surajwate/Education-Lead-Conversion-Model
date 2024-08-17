import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import argparse
import logging
from pathlib import Path
from typing import Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_folds(
    input_file: str,
    output_file: str,
    n_splits: int = 5,
    stratify: bool = False,
    target_column: str = None,
    random_state: int = 42
) -> None:
    """
    Create k-fold splits (stratified or not) and save the result.
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the output CSV file.
        n_splits (int): Number of splits for k-fold. Defaults to 5.
        stratify (bool): Whether to use stratified k-fold. Defaults to False.
        target_column (str): Name of the target column for stratification. Required if stratify is True.
        random_state (int): Random state for reproducibility. Defaults to 42.
    """
    try:
        # Read the data
        df = pd.read_csv(input_file)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")

        # Create a kfold column
        df['kfold'] = -1

        # Randomize the rows
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        if stratify:
            if target_column is None or target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found or not specified for stratified split.")
            y = df[target_column].values
            splitter: Union[KFold, StratifiedKFold] = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            split_args = {'X': df, 'y': y}
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            split_args = {'X': df}

        # Fill the new kfold column
        for fold, (_, val_idx) in enumerate(splitter.split(**split_args)):
            df.loc[val_idx, 'kfold'] = fold

        # Save the dataframe
        df.to_csv(output_file, index=False)
        logging.info(f"Folded data saved to {output_file}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create k-fold splits of a dataset.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output CSV file')
    parser.add_argument('--splits', type=int, default=5, help='Number of splits for k-fold')
    parser.add_argument('--stratify', action='store_true', help='Use stratified k-fold')
    parser.add_argument('--target', type=str, help='Name of the target column for stratification')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file {args.input} does not exist.")

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    create_folds(
        args.input,
        args.output,
        args.splits,
        args.stratify,
        args.target,
        args.random_state
    )
    
"""
Example Usage:
To create non-stratified k-fold:
python create_kfold_splits.py --input train.csv --output train_folds.csv --splits 5
To create stratified k-fold:
python create_kfold_splits.py --input train.csv --output train_folds.csv --splits 5 --stratify --target target_column_name
"""