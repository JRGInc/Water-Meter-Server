import logging
import os

# Log paths and files
LOGPATH = os.path.normpath(os.getenv('JANUSDAT_CORE_LOG_PATH',
                                     '/var/log/JanusDATA/'))  # Log path
janusdata = os.path.join(LOGPATH, 'janusdata')  # Log file
server = os.path.join(LOGPATH, 'server')  # Log file


class Logging(object):
    def __init__(self):
        self.config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'simple': {
                    'format': '%(levelname)s %(message)s',
                },
                'verbose': {
                    'format': " * %(asctime)s * %(levelname)s: " +
                              "<function '%(funcName)s' from '%(filename)s'>: %(message)s",
                },
            },
            'loggers': {
                'janusdata': {
                    'handlers': ['janusdata'],
                    'propagate': False,
                    'level': 'INFO',
                },
                'server': {
                    'handlers': ['server'],
                    'propagate': True,
                    'level': 'INFO',
                },
            },
            'handlers': {
                'janusdata': {
                    'level': 'DEBUG',
                    'formatter': 'verbose',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': janusdata,
                    'maxBytes': 8192000,
                    'backupCount': 40,
                },
                'server': {
                    'level': 'DEBUG',
                    'formatter': 'verbose',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': server,
                    'maxBytes': 8192000,
                    'backupCount': 40,
                },
                'console': {
                    'level': 'DEBUG',
                    'formatter': 'simple',
                    'class': 'logging.StreamHandler',
                },
            }
        }
