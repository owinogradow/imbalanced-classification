import logging
import logging.config

def get_logger(name):

    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('{}.log'.format(name),encoding='utf8')
    formatter = logging \
        .Formatter('[%(asctime)s]'
                    # [%(processName)s %(process)-6d]'
                    '[%(levelname)-8s][%(funcName)-20s] '
                    '%(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger

logger = get_logger('/tmp/FileListGen')