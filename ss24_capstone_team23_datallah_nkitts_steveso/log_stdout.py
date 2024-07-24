import logging
import sys
import argparse
import os

# Define default values
defaults = {
    'log_file': 'model_training.log',
    'log_level': logging.INFO,
    'log_format': '%(asctime)s - %(levelname)s - %(message)s'
}

class StreamToLogger:
    def __init__(self, **kwargs):         
        self.log_file = kwargs.get('log_file', defaults['log_file'])
        self.log_level = kwargs.get('log_level', defaults['log_level'])
        self.log_format = kwargs.get('log_format', defaults['log_format'])
        self.logger = self._initialize_logger()

    def _initialize_logger(self):
        # Initialize logging
        logger = logging.getLogger(__name__)
        logger.setLevel(self.log_level)

        if not logger.hasHandlers():
            # Create a file handler to capture the output
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(self.log_level)

            # Create a console handler to capture the output
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)

            # Create a logging format and add handlers
            formatter = logging.Formatter(self.log_format)
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        # Redirect stdout to the logger
        sys.stdout = self
        return logger

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configure logger")
    parser.add_argument('--log_file', type=str, help='Log file path')
    parser.add_argument('--log_level', type=str, help='Log level (e.g., INFO, DEBUG)')
    parser.add_argument('--log_format', type=str, help='Log format')
    args = parser.parse_args()

    # Prepare the configuration dictionary
    config = defaults.copy()
    if args.log_file:
        config['log_file'] = args.log_file
    if args.log_level:
        config['log_level'] = getattr(logging, args.log_level.upper(), logging.INFO)
    if args.log_format:
        config['log_format'] = args.log_format

    # Initialize logger with the configuration
    logger = StreamToLogger(**config)

    # Example print statement to test logging
    print("This will be logged.")
