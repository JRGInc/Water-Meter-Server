__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import logging
import os

logfile = 'januswm'
logger = logging.getLogger(logfile)


class CoreCfg(object):
    """
    Class attributes of configuration settings for capture operations
    """

    def __init__(
            self
    ) -> None:
        """
        Sets object properties directly
        """

        # Define base paths
        base_dir = os.path.dirname('/opt/Janus/')
        program_dir = 'Water-Meter-Server/'
        self.program_dir = os.path.join(
            base_dir,
            program_dir
        )

        # Define core paths
        core_dirs_dict = {
            'cfg': 'config/',
            'imgs': 'images/',
            'py3': 'python3/',
            'wgts': 'weights/'
        }

        # Define core paths
        self.core_path_dict = {
            'cfg': os.path.join(
                self.program_dir,
                core_dirs_dict['cfg']
            ),
            'imgs': os.path.join(
                self.program_dir,
                core_dirs_dict['imgs']
            ),
            'py3': os.path.join(
                self.program_dir,
                core_dirs_dict['py3']
            ),
            'wgts': os.path.join(
                self.program_dir,
                core_dirs_dict['wgts']
            )
        }

        # Define core paths
        self.img_dirs_dict = {
            'orig': '01--original/',
            'bbox': '02--bboxes/',
            'grotd': '03--grotated/',
            'frotd': '04--frotated/',
            'rect': '05--rectangled/',
            'wind': '06--windowed/',
            'inv': '07--inverted/',
            'cont': '08--contoured/',
            'digs': '09--digits/',
            'olay': '10--overlaid/'
        }

    def get(
        self,
        attrib: str
    ) -> any:
        """
        Gets configuration attributes

        :param attrib: str

        :return: any
        """
        if attrib == 'program_dir':
            return self.program_dir
        elif attrib == 'core_path_dict':
            return self.core_path_dict
        elif attrib == 'img_dirs_dict':
            return self.img_dirs_dict
