#!/usr/bin/env python3
__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

# Initialize Application processes
if __name__ == '__main__':
    import logging.config
    from server.webserver import FlaskServer
    from config.log import LogCfg
    from tendo import singleton

    # Prevent a second instance of this script from running
    me = singleton.SingleInstance()

    # Configure logging
    log_config_obj = LogCfg()
    logging.config.dictConfig(log_config_obj.config)

    logfile = 'janusserver'
    logger = logging.getLogger(logfile)
    for i in range(1, 6):
        logger.info('')

    log = 'Janus server logging started'
    logger.info(log)

    # Start Tornado webserver main and websocket application processes
    flask_obj = FlaskServer()
    flask_obj.webserver()

    log = 'Janus Server startup completed.'
    logger.info(log)
    print(log)
