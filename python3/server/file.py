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
        core_cfg = CoreCfg()
        core_path_dict = core_cfg.get(attrib='core_path_dict')
        data_dirs_dict = core_cfg.get(attrib='data_dirs_dict')

        user_agent = post_data.headers['User-Agent'].split('_')[0]
        file_name = post_data.headers['User-Agent']
        data_path = os.path.join(
            core_path_dict['data'],
            user_agent + '/',
            data_dirs_dict['errs']
        )
        file_url = os.path.join(
            data_path,
            file_name
        )

        try:
            f = open(file_url, 'w+b')
            f.write(post_data.data)
            f.close()

            log = 'Text datafile {0} successfully uploaded!'. \
                format(file_url)
            logger.info(log)
            print(log)

        except Exception as exc:
            log = 'Failed to download text datafile.'
            logger.error(msg=log)
            logger.error(msg=exc)
            print(log)
            print(exc)

        return log

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
        data_dirs_dict = core_cfg.get(attrib='data_dirs_dict')

        user_agent = post_data.headers['User-Agent'].split('_')[0]
        img_orig_name = 'orig_' + post_data.headers['User-Agent'][13::]
        data_path = os.path.join(
            core_path_dict['data'],
            user_agent + '/',
            data_dirs_dict['orig']
        )
        img_orig_url = os.path.join(
            data_path,
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
            log = 'Failed to download binary datafile.'
            logger.error(msg=log)
            logger.error(msg=exc)
            print(log)
            print(exc)

        return post_err, data_path, img_orig_name, img_orig_url, log

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
