#!/usr/bin/env python3
__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

# Initialize Application processes
if __name__ == '__main__':
    import cv2
    import logging.config
    import os
    import time
    from common import db_ops, file_ops, img_ops
    from config.core import CoreCfg
    from config.log import LogCfg
    from config.tensor import TensorCfg
    from machine.tensor import Tensor
    from tendo import singleton

    # Prevent a second instance of this script from running
    me = singleton.SingleInstance()

    # Configure logging
    log_config_obj = LogCfg()
    logging.config.dictConfig(log_config_obj.config)

    # Setup logging
    logfile = 'janusdata'
    logger = logging.getLogger(logfile)
    for i in range(1, 6):
        logger.info('')

    # Error values dictionary
    err_dict = {
        'shape': True,
        'build_yolo': True,
        'build_incept': True,
        'detection': True,
        'img_bbox': True,
        'img_angle': True,
        'img_rotd': True,
        'img_digw': True,
        'img_digs': True,
        'img_olay': True,
        'prediction': True,
    }

    # Load configuration settings
    core_cfg = CoreCfg()
    data_dir = core_cfg.get(attrib='data_dir')
    data_dirs_dict = core_cfg.get(attrib='data_dirs_dict')
    img_save_dict = core_cfg.get(attrib='img_save_dict')
    tensor_cfg = TensorCfg(core_cfg=core_cfg)
    pred_freq = tensor_cfg.get(attrib='pred_freq')
    incoming_path = os.path.join(
        data_dir,
        'incoming/',
    )

    # Setup TensorFlow models
    tensor = Tensor(core_cfg=core_cfg)
    err_dict['build_yolo'], yolo_model, yolo_classes = tensor.build_yolo_model()
    log = 'Build YOLO error: {0}'.format(err_dict['build_yolo'])
    logger.info(log)
    print(log)

    err_dict['build_incept'], incept_model = tensor.build_incept_model()
    log = 'Build Inception error: {0}'.format(err_dict['build_incept'])
    logger.info(log)
    print(log)

    # Proceed only if model setup successful
    if not err_dict['build_yolo'] and not err_dict['build_incept']:

        while True:
            # Create sorted list of image names from source directory for iteration
            incoming_names = iter(sorted(os.listdir(incoming_path)))
            for incoming_name in incoming_names:
                # Skip any file that is not a JPEG image.
                file_name, file_ext = os.path.splitext(incoming_name)
                if file_ext != '.jpg':
                    continue

                incoming_url = os.path.join(
                    incoming_path,
                    incoming_name
                )
                log = 'Processing incoming image: {0}'.format(incoming_url)
                logger.info(log)
                print(log)

                device = incoming_name.split(sep='_')[0]
                img_dtg = incoming_name.split(sep='_')[1] + '_' + \
                          incoming_name.split(sep='_')[2]
                device_path = os.path.join(
                    data_dir,
                    device + '/',
                )

                # Build image urls from image name
                img_orig_url = os.path.join(
                    device_path,
                    data_dirs_dict['orig'],
                    'orig' + incoming_name[12::]
                )

                img_bbox_url = os.path.join(
                    device_path,
                    data_dirs_dict['bbox'],
                    'bbox' + incoming_name[12::]
                )
                img_grotd_url = os.path.join(
                    device_path,
                    data_dirs_dict['grotd'],
                    'grotd' + incoming_name[12::]
                )
                img_frotd_url = os.path.join(
                    device_path,
                    data_dirs_dict['frotd'],
                    'frotd' + incoming_name[12::]
                )
                img_rect_url = os.path.join(
                    device_path,
                    data_dirs_dict['rect'],
                    'rect' + incoming_name[12::]
                )
                img_digw_url = os.path.join(
                    device_path,
                    data_dirs_dict['digw'],
                    'digw' + incoming_name[12::]
                )
                img_inv_url = os.path.join(
                    device_path,
                    data_dirs_dict['inv'],
                    'inv' + incoming_name[12::]
                )
                img_olay_url = os.path.join(
                    device_path,
                    data_dirs_dict['olay'],
                    'olay' + incoming_name[12::]
                )

                copy_err = file_ops.copy_file(
                    incoming_url,
                    img_orig_url
                )

                # Check to determine that file is legitimate
                if not copy_err and os.path.isfile(path=img_orig_url):
                    log = 'Copied image: {0}'.format(img_orig_url)
                    logger.info(log)
                    print(log)

                    # Read image into OpenCV format
                    img_orig = cv2.imread(filename=img_orig_url)

                    # Get shape, height = element 0, width = element 1
                    img_orig_shape = img_orig.shape

                    # If meets criteria, continue execution, if not skip and iterate
                    # to next image.
                    if (img_orig.shape[0] == 2464) and (img_orig.shape[1] == 3280):
                        pass
                    elif (img_orig.shape[0] == 1536) and (img_orig.shape[1] == 1536):
                        pass
                    else:
                        continue

                    # Perform object detection on image
                    err_dict['detection'], bbox_dict = tensor.detect_yolo(
                        model=yolo_model,
                        img_orig=img_orig,
                        img_orig_shape=img_orig_shape,
                        img_save_dict=img_save_dict,
                        img_bbox_url=img_bbox_url,
                        classes=yolo_classes
                    )
                    log = 'Detection error: {0}'.format(err_dict['detection'])
                    logger.info(log)
                    print(log)

                    # If object detection does not produce errors, used detected
                    # objects to find angle of rotation
                    img_ang_list = None
                    if not err_dict['detection'] and bbox_dict is not None:
                        err_dict['img_angle'], img_ang_list = img_ops.find_angles(
                            bbox_dict=bbox_dict,
                        )
                        log = 'Angle error: {0}, {1}'.format(err_dict['img_angle'], img_ang_list)
                        logger.info(log)
                        print(log)

                    # If finding angles does not produce errors, use angle
                    # to rotate image
                    img_rotd = None
                    if not err_dict['img_angle'] and img_ang_list is not None:
                        err_dict['img_rotd'], img_rotd = img_ops.rotate(
                            img_orig=img_orig,
                            img_save_dict=img_save_dict,
                            img_grotd_url=img_grotd_url,
                            img_frotd_url=img_frotd_url,
                            img_orig_shape=img_orig_shape,
                            img_ang_list=img_ang_list,
                        )
                        log = 'Rotation error: {0}'.format(err_dict['img_rotd'])
                        logger.info(log)
                        print(log)

                    # If rotation does not produce errors, crop digit window from
                    # rotated image
                    img_digw = None
                    if not err_dict['img_rotd'] and img_rotd is not None:
                        err_dict['img_digw'], img_digw = img_ops.crop_rect(
                            img_rotd=img_rotd,
                            img_save_dict=img_save_dict,
                            img_rect_url=img_rect_url,
                            img_digw_url=img_digw_url
                        )
                        log = 'Digit window error: {0}'.format(err_dict['img_digw'])
                        logger.info(log)
                        print(log)

                    # If cropping digit window does not produce errors, crop
                    # individual digits from digit window
                    img_digs = None
                    if not err_dict['img_digw'] and img_digw is not None:
                        err_dict['img_digs'], img_digs = img_ops.crop_digits(
                            img_digw=img_digw,
                            img_save_dict=img_save_dict,
                            img_inv_url=img_inv_url,
                            device_path=device_path,
                            data_dirs_dict=data_dirs_dict
                        )
                        log = 'Crop digits error: {0}'.format(err_dict['img_digs'])
                        logger.info(log)
                        print(log)

                    # Create text for overlaid image
                    img_orig_dtg = str(incoming_name).split('_')[1] + ' ' + \
                        str(incoming_name).split('_')[2]
                    img_olay_text = 'Date & Time: ' + img_orig_dtg

                    # If cropping digits does not produce errors, perform
                    # TensorFlow predictions on digits
                    if not err_dict['img_digs'] and img_digs is not None:

                        err_dict['prediction'], pred_list_pri, pred_list_alt, olay_text = \
                                tensor.predict_inception(
                            model=incept_model,
                            img_digs=img_digs,
                            img_orig_dtg=img_orig_dtg
                        )
                        img_olay_text = img_olay_text + olay_text
                        log = 'Prediction error: {0}'.format(err_dict['prediction'])
                        logger.info(log)
                        print(log)

                        if not err_dict['prediction']:
                            db_ops.store_data(
                                device=device,
                                img_dtg=img_dtg,
                                data=pred_list_alt
                            )

                    # Overlay image with date-time stamp and value if
                    # no TensorFlow error.
                    if not err_dict['prediction'] and img_digw is not None:
                        err_dict['img_olay'] = img_ops.overlay(
                            img_orig_shape=img_orig_shape,
                            img_digw=img_digw,
                            img_olay_url=img_olay_url,
                            img_olay_text=img_olay_text
                        )
                        log = 'Overlay error: {0}'.format(err_dict['img_olay'])
                        logger.info(log)
                        print(log)

                    if not err_dict['img_olay'] and os.path.isfile(path=img_olay_url):
                        os.remove(incoming_url)

                        if not os.path.isfile(path=incoming_url):
                            log = 'Image {0} successfully removed.'.format(incoming_url)
                            logger.info(msg=log)
                            print(log)
                        else:
                            log = 'Failed to remove image {0}d.'.format(incoming_url)
                            logger.error(msg=log)
                            print(log)

                # If file is illegitimate, log error, move to next file
                else:
                    img_ang_err = True
                    log = 'Failed to locate image {0} to process.'. \
                        format(img_orig_url)
                    logger.error(msg=log)
                    print(log)

            time.sleep(pred_freq)

    else:
        log = 'Unable to begin image processing due to errors building TensorFlow models.'
        logger.error(log)
        print(log)
