import os
import time

_LOG_FORMAT = '%s %s \r\n'
_LOG_FILE_PATH = 'logs'

if not os.path.isdir(_LOG_FILE_PATH):
    os.mkdir(_LOG_FILE_PATH)

def log(log_file_name: str,log_content):
    now_date = time.strftime("%Y-%m-%d", time.localtime())
    with open(f'{_LOG_FILE_PATH}/{log_file_name}-{now_date}.log', 'a') as log_file:
        date_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_file.writelines(_LOG_FORMAT % (date_time, str(log_content)))
