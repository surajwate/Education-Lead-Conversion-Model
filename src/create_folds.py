import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import argparse
import logging
from pathlib import Path
from typing import Union

"""
Example Usage:

Non-Stratified K-Fold:

1. With Default Split Value (5 splits):
   - Full Argument Names:
     python create_folds.py --input train.csv --output train_folds.csv
   - Shortcut Argument Names:
     python create_folds.py -i train.csv -o train_folds.csv

2. Specifying Number of Splits:
   - Full Argument Names:
     python create_folds.py --input train.csv --output train_folds.csv --splits 5
   - Shortcut Argument Names:
     python create_folds.py -i train.csv -o train_folds.csv -s 5

Stratified K-Fold:

1. With Default Split Value (5 splits):
   - Full Argument Names:
     python create_folds.py --input train.csv --output train_folds.csv --stratify --target target_column_name
   - Shortcut Argument Names:
     python create_folds.py -i train.csv -o train_folds.csv -S -t target_column_name

2. Specifying Number of Splits:
   - Full Argument Names:
     python create_folds.py --input train.csv --output train_folds.csv --splits 5 --stratify --target target_column_name
   - Shortcut Argument Names:
     python create_folds.py -i train.csv -o train_folds.csv -s 5 -S -t target_column_name

Summary:
- Default Values: If you don't specify the --splits or -s option, the script will default to 5 splits.
- Stratified K-Fold: To create a stratified k-fold, use the --stratify or -S option along with specifying the target column using --target or -t.
- Flexibility: You can use either the full argument names or the shortcut versions depending on your preference.
"""

# Configure logging
log_file_name = "create_folds.log"
def configure_logging(log_file_name):
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / log_file_name,
        level=logging.INFO,  # Consider DEBUG for more detailed output
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",
    )


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
        # Read the data
        df = pd.read_csv(input_path)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")

        # Create a kfold column
        df["kfold"] = -1

        # Randomize the rows
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        if stratify:
            if target_column is None or target_column not in df.columns:
                raise ValueError(
                    f"Target column '{target_column}' not found or not specified "
                    f"for stratified split."
                )
            y = df[target_column].values
            splitter: Union[KFold, StratifiedKFold] = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            split_args = {"X": df, "y": y}
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            split_args = {"X": df}

        # Fill the new kfold column
        for fold, (_, val_idx) in enumerate(splitter.split(**split_args)):
            df.loc[val_idx, "kfold"] = fold

        # Save the dataframe
        df.to_csv(output_path, index=False)
        logging.info(f"Folded data saved to {output_path}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create k-fold splits of a dataset.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help='Filename of the input CSV file within the "input/" directory',
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help='Filename to save the output CSV file within the "input/" directory',
    )
    parser.add_argument(
        "-s", "--splits", type=int, default=5, help="Number of splits for k-fold"
    )
    parser.add_argument(
        "-S", "--stratify", action="store_true", help="Use stratified k-fold"
    )
    parser.add_argument(
        "-t", "--target", type=str, help="Name of the target column for stratification"
    )
    parser.add_argument(
        "-r",
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility",
    )

    args = parser.parse_args()

    input_path = Path(f"./input/{args.input}")
    output_path = Path(f"./input/{args.output}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist.")

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    create_folds(
        input_path,
        output_path,
        args.splits,
        args.stratify,
        args.target,
        args.random_state,
    )
    