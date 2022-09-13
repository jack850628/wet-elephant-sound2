#!/bin/bash

set -a #讓config.env可以成功寫入環境變數
source config.env
service mariadb start
mysql <<EOF
create database PikapiAndDasiGashaponMachine;
create user '$DATABASE_USER_NAME'@'localhost' identified by '$DATABASE_USER_PASSWORD';
grant all privileges on *.* to '$DATABASE_USER_NAME'@'localhost';
flush privileges;
EOF
mkdir /app/logs/nginx/
touch /app/logs/nginx/access.log
touch /app/logs/nginx/error.log
service nginx start
uwsgi --ini /app/uwsgi.ini