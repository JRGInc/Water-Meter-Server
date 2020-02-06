import json
import logging
from flask import json, request
from server.file import PostFile

__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

logfile = 'server'
logger = logging.getLogger(logfile)


class HandlerApp(
    object
):
    def __init__(
        self,
        flask_app
    ):
        self.flask_app = flask_app
        self.post_file = PostFile()

    def handler(
        self
    ):

        @self.flask_app.route('/upload', methods=['POST'])
        def api_message():
            log = "Incoming transmission!"
            logger.info(log)
            logger.info(request.headers)
            print(log)
            print(request.headers)
            # print(request.data)
            print("\n\n")

            if request.headers['Content-Type'] == 'application/json':
                logger.info(request.data)
                result = self.post_file.post_json(request)
                return result

            # Receive plain text files
            elif request.headers['Content-Type'] == 'text/plain':
                result = self.post_file.post_text(request)
                return result

            elif request.headers['Content-Type'] == 'application/octet-stream':
                result = self.post_file.post_binary(request)
                print(result)
                return result

            else:
                return '415 Unsupported Media Type'
