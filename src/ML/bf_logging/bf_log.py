import os
import logging
from logging.handlers import RotatingFileHandler

# TODO: Move log configuration to a file
def setup_logger():
    """ Sets up logger.
    """

    # Logger will not be set up twice.
    if logging.getLogger('').handlers:
        return

    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG)

    # define File handler
    file_handler = RotatingFileHandler(filename='logs/barefoot_ml.log', maxBytes=512000, backupCount=9)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s: %(message)s'))
    logging.getLogger('').addHandler(file_handler)

    # define a Handler which writes INFO messages or higher to the console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s - %(levelname)-8s - %(funcName)s: %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
