import cv2
import logging
import os
from config.core import CoreCfg
from flask import request
from machine import prediction
from machine.tensor import Tensor
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
        core_cfg = CoreCfg()

        self.flask_app = flask_app
        self.post_file = PostFile()

        # Setup TensorFlow object
        self.tensor = Tensor(core_cfg=core_cfg)

        # Build YOLO v3 model and retrieve classes
        self.err_yolo, self.yolo_model, self.yolo_classes = \
            self.tensor.build_yolo_model()
        if self.err_yolo:
            log = 'Unable to build YOLO v3 model'
            logger.error(log)
            print(log)
        else:
            log = 'YOLO v3 model successfully built.'
            logger.info(log)
            print(log)

        # Build Inception v4 model
        self.err_incept, self.incept_model = self.tensor.build_incept_model()
        if self.err_incept:
            log = 'Unable to build Inception v4 model'
            logger.error(log)
            print(log)
        else:
            log = 'Inception v4 model successfully built.'
            logger.info(log)
            print(log)

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
                post_err, img_path, img_orig_name, img_orig_url, result = \
                    self.post_file.post_binary(request)

                # Need to ensure that no errors were thrown and that the image
                # file is valid prior to processing image,
                if not self.err_yolo and not self.err_incept and not post_err \
                        and os.path.isfile(path=img_orig_url):
                    img_orig = cv2.imread(filename=img_orig_url)
                    prediction.process_image(
                        tensor_obj=self.tensor,
                        yolo_model=self.yolo_model,
                        yolo_classes=self.yolo_classes,
                        incept_model=self.incept_model,
                        img_orig=img_orig,
                        img_path=img_path,
                        img_orig_name=img_orig_name
                    )
                print(result)
                return result

            else:
                return '415 Unsupported Media Type'
