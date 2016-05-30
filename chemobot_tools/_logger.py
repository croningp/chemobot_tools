import logging

__logger_root_name__ = 'chemobot_tools'


def create_logger(name):
    return logging.getLogger(__logger_root_name__ + '.' + name)
