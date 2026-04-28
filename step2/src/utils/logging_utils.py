import logging
import sys
from pathlib import Path


def setup_logging(name: str = "pipeline", log_file: str = "pipeline.log") -> logging.Logger:
    root_path = Path(__file__).parents[3]
    log_dir = root_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / log_file

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


logger = setup_logging()
