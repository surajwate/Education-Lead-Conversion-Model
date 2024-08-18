import logging
from pathlib import Path

def configure_logging(log_file_name="pipeline.log"):
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create a custom logger for the module
    logger = logging.getLogger(log_file_name)
    
    # Avoid adding multiple handlers if the logger already has handlers
    if not logger.handlers:
        file_handler = logging.FileHandler(log_dir / log_file_name)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    return logger
