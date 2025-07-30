import os
import logging
from config import settings

# Ensure the log directory exists
PATH = settings.log_path
os.makedirs(PATH, exist_ok=True)  # Prevent FileExistsError

# Define log file paths for each level
LOG_FILE_PATHS = {
    "debug": f"{PATH}/debug.log",
    "info": f"{PATH}/info.log",
    "warning": f"{PATH}/warning.log",
    "error": f"{PATH}/error.log",
    "critical": f"{PATH}/critical.log"
}


# Create a function to configure a file handler for a specific level
def create_file_handler(level, log_file):
    handler = logging.FileHandler(log_file)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    return handler


# Configure the logger
logger = logging.getLogger("sa_fastapi")
logger.setLevel(logging.DEBUG)  # Capture all levels

# Add a stream handler for console output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Add file handlers for each log level
for level_name, log_file in LOG_FILE_PATHS.items():
    level = getattr(logging, level_name.upper())  # Convert "debug" to logging.DEBUG, etc.
    file_handler = create_file_handler(level, log_file)

    # Add a filter to only log messages of this level
    file_handler.addFilter(lambda record, lvl=level: record.levelno == lvl)

    logger.addHandler(file_handler)

# if __name__ == '__main__':
#     # Log messages sample
#     logger.debug("This is a debug message")
#     logger.info("This is an info message")
#     logger.warning("This is a warning message")
#     logger.error("This is an error message")
#     logger.critical("This is a critical message")
