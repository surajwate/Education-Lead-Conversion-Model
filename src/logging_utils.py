import logging
from pathlib import Path

def configure_logging(log_file_name="pipeline.log"):
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / log_file_name,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",  # Use 'w' to overwrite the log file on each run, 'a' to append
    )
    logging.info("Logging is configured.")
