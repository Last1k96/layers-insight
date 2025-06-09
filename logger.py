import logging
import colorlog

def setup_logger():
    """Set up and configure a colored logger."""
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s[%(levelname)s] %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))

    logger = logging.getLogger('layers-insight')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


# Initialize the global logger
log = setup_logger()