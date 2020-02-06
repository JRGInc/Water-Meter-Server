import json
import logging
import os
from datetime import datetime
from flask import json as flask_json

__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

logfile = 'server'
logger = logging.getLogger(logfile)


class PostFile(
    object
):
    def post_json(
        self,
        post_data
    ):
        """
        POST handler

        :param post_data: request object
        """
        result = 'Unknown JSON request.'
        json_str = flask_json.dumps(post_data.json)
        file_dict = json.loads(json_str)
        print(file_dict)
        if 'config' in file_dict:
            result = self.upload_config(
                config=file_dict['config'],
                device=post_data.headers['User-Agent']
            )

        logger.info(result)
        return result

    @staticmethod
    def post_text(
        post_data
    ):
        """
        POST handler

        :param post_data: request object
        """
        user_agent = post_data.headers['User-Agent'].split('_')[0]
        file_name = post_data.headers['User-Agent']
        print(file_name)
        file_path = '/opt/Janus/datafiles/' + user_agent + '/text/'
        result = 'Text datafile uploaded!'

        f = open(file_path + file_name, 'w+b')
        f.write(post_data.data)
        f.close()

        logger.info(result)
        return result

    @staticmethod
    def post_binary(
        post_data
    ):
        """
        POST handler

        :param post_data: request object
        """
        user_agent = post_data.headers['User-Agent'].split('_')[0]
        file_name = post_data.headers['User-Agent']
        print(file_name)
        file_path = '/opt/Janus/datafiles/' + user_agent + '/binary/'
        print(file_path)
        result = 'Binary datafile uploaded!'

        f = open(file_path + file_name, 'w+b')
        f.write(post_data.data)
        f.close()

        logger.info(result)
        return result

    @staticmethod
    def upload_config(
        config: str,
        device: str
    ):
        """
        POST handler

        :param config: str
        :param device: str

        :return result: str
        """
        file_url = '/opt/Janus/datafiles/' + device + '/upload/' + config + '.ini'

        if os.path.isfile(path=file_url):
            f = open(file_url, 'r')
            result = f.read()
            f.close()

        else:
            result = 'Incorrect config file request.'

        return result
