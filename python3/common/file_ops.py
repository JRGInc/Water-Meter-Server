__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import csv
import logging
from os import close, remove
from shutil import move
from tempfile import mkstemp

logfile = 'januswm'
logger = logging.getLogger(logfile)


def f_request(
    file_cmd: str,
    file_name: str,
    data_file_in: list = None,
    data_loc: int = 0,
    num_bytes: int = 0,
    num_lines: int = 0,
    search_field: str = '',
    replace_line: str = ''
) -> [str, int]:
    """
    Executes file transaction

    :param file_cmd: str
    :param file_name: str
    :param data_loc: int
    :param num_bytes: int
    :param num_lines: int
    :param data_file_in: str
    :param search_field: str
    :param replace_line: str

    :return read_err: bool
    :return data: 0 (read_err = True)
    :return data: str (read_err = False)
    """
    data_file_out = ''
    file_mode = ''
    temp_path = None
    file_hdlr = None
    file_temp = None
    log_oserror = ''
    log_no_oserror = ''
    line_count = 0

    # Set file mode dependent upon required command, build log entries
    if file_cmd == 'data_read':
        file_mode = 'r'
        log_no_oserror = 'Data retrieval from file {0} succeeded.'.format(file_name)
        log_oserror = 'Data retrieval from file {0} failed.'.format(file_name)

    elif file_cmd == 'data_line_read':
        file_mode = 'r'
        log_no_oserror = 'Data line retrieval from file {0} succeeded.'.format(file_name)
        log_oserror = 'Data line retrieval from file {0} failed.'.format(file_name)

    elif file_cmd == 'fld_read':
        file_mode = 'r'
        log_no_oserror = 'Field retrieval from file {0} succeeded.'.format(file_name)
        log_oserror = 'Field retrieval from file {0} failed.'.format(file_name)

    elif file_cmd == 'fld_edit':
        file_mode = 'r'
        log_no_oserror = 'Field update in file {0} succeeded.'.format(file_name)
        log_oserror = 'Field update in file {0} failed.'.format(file_name)

    elif file_cmd == 'file_replace':
        file_mode = 'w'
        log_no_oserror = 'Content replacement in file {0} succeeded.'.format(file_name)
        log_oserror = 'Content replacement in file {0} failed.'.format(file_name)

    elif file_cmd == 'file_line_write':
        file_mode = 'w'
        log_no_oserror = 'Content write in file {0} succeeded.'.format(file_name)
        log_oserror = 'Content write in file {0} failed.'.format(file_name)

    elif file_cmd == 'file_line_append':
        file_mode = 'a'
        log_no_oserror = 'Content append in file {0} succeeded.'.format(file_name)
        log_oserror = 'Content append in file {0} failed.'.format(file_name)

    elif file_cmd == 'file_line_fifo':
        file_mode = 'r'
        log_no_oserror = 'Content replacement in file {0} succeeded.'.format(file_name)
        log_oserror = 'Content replacement in file {0} failed.'.format(file_name)

    elif file_cmd == 'file_csv_appendlist':
        file_mode = 'a'
        log_no_oserror = 'Content replacement in file {0} succeeded.'.format(file_name)
        log_oserror = 'Content replacement in file {0} failed.'.format(file_name)

    elif file_cmd == 'file_csv_writelist':
        file_mode = 'w'
        log_no_oserror = 'Content replacement in file {0} succeeded.'.format(file_name)
        log_oserror = 'Content replacement in file {0} failed.'.format(file_name)

    try:
        # Open file in specified mode, utf-8 encoding
        file_open = open(
            file=file_name,
            mode=file_mode,
            encoding='utf-8'
        )

        # Perform follow-on file operations after opening file
        if file_cmd == 'data_read':
            file_open.seek(data_loc)
            if num_bytes > 0:
                data_file_out = file_open.read(num_bytes)
            else:
                data_file_out = file_open.read()

        elif file_cmd == 'data_line_read':
            data_file_out = file_open.read().splitlines(keepends=True)

        elif file_cmd == 'fld_read':
            for line in file_open:
                line = line.split(sep='\n')[0]
                key = line.split(sep='=')[0]
                if key == search_field:
                    data_file_out = line.split(sep='=')[1]

        elif file_cmd == 'fld_edit':
            file_hdlr, temp_path = mkstemp()
            file_temp = open(
                file=temp_path,
                mode='w',
                encoding='utf-8'
            )

            found_field = False
            for line in file_temp:
                if search_field in line:
                    file_temp.write(replace_line + '\n')
                    found_field = True

                else:
                    file_temp.write(line)

            if not found_field:
                file_temp.write(replace_line)

            file_temp.flush()

        elif file_cmd == 'file_replace':
            file_open.seek(0)
            file_open.write(data_file_in[0])
            file_open.flush()

        elif file_cmd == 'file_line_write':
            file_open.write(data_file_in[0])
            file_open.flush()

        elif file_cmd == 'file_line_append':
            file_open.write(data_file_in[0])
            file_open.flush()

        elif file_cmd == 'file_line_fifo':
            data_file_out = file_open.read().splitlines(keepends=True)
            line_count = len(data_file_out)

        elif file_cmd == 'file_csv_appendlist':
            pred_file_writer = csv.writer(
                file_open,
                delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL
            )
            pred_file_writer.writerow(data_file_in)

        elif file_cmd == 'file_csv_writelist':
            pred_file_writer = csv.writer(
                file_open,
                delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL
            )
            pred_file_writer.writerow(data_file_in)

        # Close file
        file_open.close()

        # Perform operations after closing file
        if file_cmd == 'fld_edit':
            remove(file_name)
            move(
                src=temp_path,
                dst=file_name
            )
            file_temp.close()
            close(file_hdlr)

        elif file_cmd == 'file_line_fifo':
            if line_count > num_lines:
                with open(file=file_name, mode='w') as file_open:
                    line_start = line_count - num_lines
                    file_open.writelines(data_file_out[line_start:])
            data_file_out = ''

        logger.debug(msg=log_no_oserror)

    except OSError:
        logger.exception(msg=log_oserror)

    return data_file_out
