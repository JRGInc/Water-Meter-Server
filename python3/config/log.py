__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import os

# Log paths and files
LOGPATH = os.path.normpath('/var/log/JanusDATA/')
januswm = os.path.join(LOGPATH, 'januswm')          # Log file
janusdata = os.path.join(LOGPATH, 'janusdata')      # Log file
server = os.path.join(LOGPATH, 'server')            # Log file


class LogCfg(object):
    def __init__(
        self
    ) -> None:
        """
        Instantiates logging object and sets log configuration
        """
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
                'januswm': {
                    'handlers': ['januswm'],
                    'propagate': False,
                    'level': 'INFO',
                },
                'janusdata': {
                    'handlers': ['janusdata'],
                    'propagate': False,
                    'level': 'INFO',
                },
                'server': {
                    'handlers': ['server'],
                    'propagate': True,
                    'level': 'INFO',
                }
            },
            'handlers': {
                'januswm': {
                    'level': 'DEBUG',
                    'formatter': 'verbose',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': januswm,
                    'maxBytes': 4096000,
                    'backupCount': 100,
                },
                'janusdata': {
                    'level': 'DEBUG',
                    'formatter': 'verbose',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': janusdata,
                    'maxBytes': 8192000,
                    'backupCount': 100,
                },
                'server': {
                    'level': 'DEBUG',
                    'formatter': 'verbose',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': server,
                    'maxBytes': 8192000,
                    'backupCount': 100,
                }
            }
        }
