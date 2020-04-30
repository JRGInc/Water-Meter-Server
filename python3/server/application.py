import logging.config
from common import db_ops
from config.log import LogCfg
from flask import request
from server.file import PostFile

__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'


class HandlerApp(
    object
):
    def __init__(
        self,
        flask_app
    ):
        self.flask_app = flask_app
        self.post_file = PostFile()

        # Configure logging
        log_config_obj = LogCfg()
        logging.config.dictConfig(log_config_obj.config)

        logfile = 'janusserver'
        self.logger = logging.getLogger(logfile)

    def handler(
        self
    ):

        @self.flask_app.route('/upload', methods=['POST'])
        def api_message():
            log = "Incoming transmission!"
            self.logger.info(log)
            self.logger.info(request.headers['Content-Type'])
            self.logger.info(request.headers['User-Agent'])
            self.logger.info(request.headers['Content-Length'])
            print(log)
            print(request.headers)
            # print(request.data)
            print("\n\n")

            user_agent = request.headers['User-Agent'].split('_')[0]
            db_ops.store_status([user_agent])

            if request.headers['Content-Type'] == 'application/json':
                self.logger.info(request.data)
                result = self.post_file.post_json(request)
                return result

            # Receive plain text files
            elif request.headers['Content-Type'] == 'text/plain':
                result = self.post_file.post_text(request)
                return result

            elif request.headers['Content-Type'] == 'application/octet-stream':
                result = self.post_file.post_binary(request)
                return result

            else:
                return '415 Unsupported Media Type'
