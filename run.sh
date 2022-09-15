#!/bin/bash

set -a #讓config.env可以成功寫入環境變數
source config.env
mkdir /app/logs/nginx/
touch /app/logs/nginx/access.log
touch /app/logs/nginx/error.log
cron
service nginx start
uwsgi --ini /app/uwsgi.ini