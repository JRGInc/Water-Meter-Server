__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import os

# Log paths and files
LOGPATH = os.path.normpath('/var/log/Janus/')
janus_data = os.path.join(LOGPATH, 'janusdata')          # Log file
janus_server = os.path.join(LOGPATH, 'janusserver')      # Log file


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
                'janusdata': {
                    'handlers': ['janusdata'],
                    'propagate': False,
                    'level': 'INFO',
                },
                'janusserver': {
                    'handlers': ['janusserver'],
                    'propagate': False,
                    'level': 'INFO',
                }
            },
            'handlers': {
                'janusdata': {
                    'level': 'DEBUG',
                    'formatter': 'verbose',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': janus_data,
                    'maxBytes': 8192000,
                    'backupCount': 40,
                },
                'janusserver': {
                    'level': 'DEBUG',
                    'formatter': 'verbose',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': janus_server,
                    'maxBytes': 8192000,
                    'backupCount': 40,
                }
            }
        }
