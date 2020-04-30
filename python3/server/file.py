import json
import logging
import os
from config.core import CoreCfg
from flask import json as flask_json


__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

logfile = 'janusserver'
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
        core_cfg = CoreCfg()
        data_dir = core_cfg.get(attrib='data_dir')
        data_dirs_dict = core_cfg.get(attrib='data_dirs_dict')

        user_agent = post_data.headers['User-Agent'].split('_')[0]
        file_type = post_data.headers['User-Agent'].split('_')[1]
        file_name = post_data.headers['User-Agent']

        file_url = ''
        if file_type == 'errs':
            file_url = os.path.join(
                data_dir,
                user_agent,
                data_dirs_dict['errs'],
                file_name
            )
        elif file_type == 'logs':
            file_url = os.path.join(
                data_dir,
                user_agent,
                data_dirs_dict['logs'],
                file_name
            )

        f = open(file_url, 'w+b')
        f.write(post_data.data)
        f.close()

        log = 'Text datafile {0} uploaded!'.format(file_url)
        logger.info(log)
        print(log)

        return log

    @staticmethod
    def post_binary(
        post_data
    ):
        """
        POST handler

        :param post_data: request object
        """
        core_cfg = CoreCfg()
        data_dir = core_cfg.get(attrib='data_dir')

        file_name = post_data.headers['User-Agent']
        file_url = os.path.join(
            data_dir,
            'incoming/',
            file_name
        )

        f = open(file_url, 'w+b')
        f.write(post_data.data)
        f.close()

        log = 'Binary datafile {0} uploaded!'.format(file_url)
        logger.info(log)
        print(log)

        return log

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
        core_cfg = CoreCfg()
        data_dir = core_cfg.get(attrib='data_dir')
        data_dirs_dict = core_cfg.get(attrib='data_dirs_dict')

        file_url = os.path.join(
            data_dir,
            device,
            data_dirs_dict['upld'],
            config + '.ini'
        )

        if os.path.isfile(path=file_url):
            f = open(file_url, 'r')
            result = f.read()
            f.close()

        else:
            result = 'Incorrect config file request.'
            logger.info(result)
            print(result)

        return result
