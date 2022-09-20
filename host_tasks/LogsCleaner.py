#!/usr/bin/env python3
"""
切片與清理Log檔
從 https://calvin.me/python-backup-cleaner-script/ 修改而來
使用方法: python3 host_tasks/LogsCleaner.py -p "/home/jack/文件/PerspectiveBox/logs" -k 5
"""
import sys, os
sys.path.append(os.getcwd())

import argparse
import glob
import shutil
import time
import sys
import JTools as JTools

SLICE_LOGS = [
    'uwsgi.log',
    'celery.log',
    '/nginx/access.log',
    '/nginx/error.log'
]

CLENR_LOGS = [
    'uwsgi.log*',
    'celery.log*',
    'LogsCleaner.py*',
    'nginx/access.log*',
    'nginx/error.log*',
]

def fileSlice(path: str, log_files: list):
    for log_file in log_files:
        try:
            if os.path.isfile(path + log_file) and os.path.getsize(path + log_file) > 0:
                os.system(f'cp {path + log_file} {path + log_file}.{time.strftime("%Y-%m-%d", time.localtime())}.log')
                os.system(f'echo -n "" > {path + log_file}')
        except Exception as e:
            JTools.log(os.path.basename(__file__), 'slice error: ' + e.__str__())

def clear(args, log_files: list):
    for log_file in log_files:
        BACKUPS = glob.glob(args.path + "/" + log_file) # search

        if not BACKUPS:
            JTools.log(os.path.basename(__file__), "Could not find any matching files or folders: " + args.path + "/" + log_file)
            # exit(-1)
            continue

        BACKUPS.sort(key=os.path.getctime, reverse=True) # sort by date created
        BACKUPS = [os.path.abspath(BACKUP) for BACKUP in BACKUPS] # get absolute path for all

        BACKUPS_TO_KEEP = BACKUPS[:args.keep]
        BACKUPS_TO_DELETE = BACKUPS[args.keep:]

        for BACKUP in BACKUPS_TO_KEEP:
            JTools.log(os.path.basename(__file__), "Keeping {}".format(BACKUP))

        for BACKUP in BACKUPS_TO_DELETE:
            try:
                JTools.log(os.path.basename(__file__), "Removing {}".format(BACKUP))
                if os.path.isfile(BACKUP):
                    os.remove(BACKUP)
                elif os.path.isdir(BACKUP):
                    shutil.rmtree(BACKUP)
                else:
                    raise OSError("Not a file or directory")
            except OSError as error:
                JTools.log(os.path.basename(__file__), "Error: {}".format(error.__str__()))

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Delete old backups")
    PARSER.add_argument("-p", "--path", required=True, type=str, help="path where backups are stored")
    # PARSER.add_argument("-r", "--regex", required=True, type=str, help="pattern of backup folders to match")
    PARSER.add_argument("-k", "--keep", required=True, help="how many backups to keep", default=5, type=int,)
    ARGS = PARSER.parse_args()

    fileSlice(ARGS.path, SLICE_LOGS)
    clear(ARGS, CLENR_LOGS)