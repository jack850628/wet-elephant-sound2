sudo service redis-server start
celery -A tasks worker --loglevel=info --pool=threads
set -a
source config.env 