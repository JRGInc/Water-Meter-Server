__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import os


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
        self.base_dir = os.path.dirname('/opt/Janus/DATA/')
        self.data_dir = os.path.dirname('/opt/Janus/datafiles/')

        # Define core paths
        core_dirs_dict = {
            'cfg': 'config/',
            'py3': 'python3/',
            'wgts': 'weights/'
        }

        # Define core paths
        self.core_path_dict = {
            'cfg': os.path.join(
                self.base_dir,
                core_dirs_dict['cfg']
            ),
            'py3': os.path.join(
                self.base_dir,
                core_dirs_dict['py3']
            ),
            'wgts': os.path.join(
                self.base_dir,
                core_dirs_dict['wgts']
            )
        }

        # Define image paths
        self.data_dirs_dict = {
            'errs': '00--errors/',
            'logs': '00--logs',
            'upld': '00--uploads/',
            'orig': '01--original/',
            'bbox': '02--bboxes/',
            'grotd': '03--grotated/',
            'frotd': '04--frotated/',
            'rect': '05--rectangled/',
            'digw': '06--windowed/',
            'inv': '07--inverted',
            'cont': '08--contoured/',
            'digs': '09--digits/',
            'olay': '10--overlaid/'
        }

        # Define image paths
        self.img_save_dict = {
            'bbox': True,
            'grotd': True,
            'frotd': True,
            'rect': True,
            'digw': True,
            'inv': True,
            'cont': True,
            'digs': True,
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
        if attrib == 'base_dir':
            return self.base_dir
        elif attrib == 'data_dir':
            return self.data_dir
        elif attrib == 'core_path_dict':
            return self.core_path_dict
        elif attrib == 'data_dirs_dict':
            return self.data_dirs_dict
        elif attrib == 'img_save_dict':
            return self.img_save_dict
