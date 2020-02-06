import logging
import multiprocessing
from gevent.pywsgi import WSGIServer
from flask import Flask
from server.application import HandlerApp

__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

logfile = 'server'
logger = logging.getLogger(logfile)


class FlaskServer(
    object
):
    def __init__(
        self
    ):
        """
        Setup Flask properties
        """
        self.mp_app = None
        self.mp_ws = None

        self.flask_app = Flask(
            __name__,
        )
        self.flask_app.debug = True
        self.handler_app = HandlerApp(self.flask_app)
        self.handler_app.handler()

    def application(
        self
    ):
        """
        Run main flask application, SSL handled by nginx
        """
        http_server_ws = WSGIServer(
            ('127.0.0.1',
             8889),
            self.flask_app
        )
        http_server_ws.serve_forever()

    def webserver(
        self
    ):
        """
        Call main and webserver applications via multiprocessing
        """
        # Start and run web server main listener
        try:
            self.mp_app = multiprocessing.Process(
                target=self.application,
                args=()
            )
            self.mp_app.start()
            log = 'Webserver application started.'
            logger.info(log)

        except multiprocessing.ProcessError:
            log = 'Can not start main webserver due to multiprocessing error.'
            logger.exception(log)
