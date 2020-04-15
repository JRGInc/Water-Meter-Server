import json
import logging
import os
from config.core import CoreCfg
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
    ) -> (bool, str, str, str):
        """
        POST handler

        :param post_data: request object
        """
        post_err = False

        # Load configuration settings
        core_cfg = CoreCfg()
        core_path_dict = core_cfg.get(attrib='core_path_dict')

        user_agent = post_data.headers['User-Agent'].split('_')[0]
        img_orig_name = 'orig_' + post_data.headers['User-Agent'][13::]
        img_path = os.path.join(
            core_path_dict['imgs'],
            user_agent + '/'
        )
        img_orig_url = os.path.join(
            img_path,
            img_orig_name
        )

        try:
            f = open(img_orig_url, 'w+b')
            f.write(post_data.data)
            f.close()

            log = 'Binary datafile {0} successfully uploaded!'.\
                format(img_orig_url)
            logger.info(log)
            print(log)

        except Exception as exc:
            post_err = True
            log = 'OpenCV failed to open downloaded image.'
            logger.error(msg=log)
            logger.error(msg=exc)
            print(log)
            print(exc)

        return post_err, img_path, img_orig_name, img_orig_url, log

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
