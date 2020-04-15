#!/usr/bin/env python3
__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

# Initialize Application processes
if __name__ == '__main__':
    import logging.config
    from server.webserver import FlaskServer
    from config.log import LogCfg
    from tendo import singleton

    # Configure logging
    log_config_obj = LogCfg()
    logging.config.dictConfig(log_config_obj.config)

    # Prevent a second instance of this script from running
    me = singleton.SingleInstance()

    # Inject initial empty log entries for easy-to-spot visual markers in
    # logs to show where JanusESS was started/restarted
    logs = [
        'janusdata',
        'januswm',
        'server'
    ]

    for log_file in logs:
        logger = logging.getLogger(log_file)
        for i in range(1, 6):
            logger.info('')

        log = 'Logging started'
        logger.info(log)

    # Set log file for JanusESS start sequence actions
    logfile = 'janusdata'
    logger = logging.getLogger(logfile)

    # Start Tornado webserver main and websocket application processes
    flask_obj = FlaskServer()
    flask_obj.webserver()

    log = 'JanusDATA startup completed.'
    logger.info(log)
    print(log)
