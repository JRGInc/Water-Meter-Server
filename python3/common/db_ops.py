import logging
from datetime import datetime, timezone
from influxdb import InfluxDBClient
from influxdb.exceptions import InfluxDBClientError


__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'


def store_status(
        data: list,
):
    """
    Stores module poll data in CouchDB

    :param data: list
    """
    logfile = 'janusserver'
    logger = logging.getLogger(logfile)

    data_idb_in = [
        {
            'measurement': 'status',
            'tags': {
                'device': str(data[0])
            },
            'time': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S'),
            'fields': {
                'value': 1
            }
        }
    ]

    # Store data packet to local InfluxDB JanusESS database
    try:
        influxdb_client = InfluxDBClient(
            host='localhost',
            port=8086,
            database='jrgwm_status',
            username='jrgwm',
            password='AluetianTem',
            timeout=5
        )
        influxdb_result = influxdb_client.write_points(data_idb_in)

        if influxdb_result:
            log = 'Upload of heartbeat data for device {0} '.format(data[0]) +\
                  'to InfluxDB successful.'
            logger.info(log)

        else:
            log = 'Could not upload heartbeat data device {0} '.format(data[0]) + \
                  'to InfluxDB due to write points error.'
            logger.warning(log)

    except InfluxDBClientError as exc:
        log = 'InfluxDB server responded with error.'
        logger.error(msg=log)
        logger.error(msg=exc)


def store_data(
        device: str,
        img_dtg: str,
        data: list,
):
    """
    Stores module poll data in CouchDB

    :param device: str
    :param img_dtg: str
    :param data: list
    """
    logfile = 'janusdata'
    logger = logging.getLogger(logfile)

    datetime_object = datetime.strptime(img_dtg, '%Y-%m-%d_%H%M')
    data_idb_in = [
        {
            'measurement': 'digits',
            'tags': {
                'device': device
            },
            'time': datetime_object.astimezone().astimezone(timezone.utc).
            replace(tzinfo=None).strftime('%Y-%m-%dT%H:%M'),
            'fields': {
                'value': int(data[1])
            }
        }
    ]

    # Store data packet to local InfluxDB JanusESS database
    try:
        influxdb_client = InfluxDBClient(
            host='localhost',
            port=8086,
            database='jrgwm_data',
            username='jrgwm',
            password='AluetianTem',
            timeout=5
        )
        influxdb_result = influxdb_client.write_points(data_idb_in)

        if influxdb_result:
            log = 'Upload of heartbeat data for device {0} '.format(data[0]) + \
                  'to InfluxDB successful.'
            logger.info(log)

        else:
            log = 'Could not upload heartbeat data device {0} '.format(data[0]) + \
                  'to InfluxDB due to write points error.'
            logger.warning(log)

    except InfluxDBClientError as exc:
        log = 'InfluxDB server responded with error.'
        logger.error(msg=log)
        logger.error(msg=exc)
