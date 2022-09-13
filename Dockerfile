FROM tensorflow/tensorflow:2.9.1

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app


RUN apt-get update
RUN apt-get install -y nginx vim libgl1-mesa-glx cron ffmpeg
RUN apt-get clean && \
    apt-get autoremove
	
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /app

RUN /bin/cp -f /app/nginx_website.conf /etc/nginx/sites-available/default
RUN mkdir /var/log/uwsgi && touch /var/log/uwsgi/uwsgi.log
RUN crontab /app/cronfile

#CMD ["gunicorn", "-k", "gevent", "-b", "0.0.0.0:9000", "app:app"]

#CMD ["service", "nginx", "start"]

#CMD service nginx start ; python3 App.py
#CMD service nginx start ; service redis-server start ; uwsgi --ini /app/uwsgi.ini

CMD ["cron", "start"]
CMD ["/bin/bash", "-c", "source /app/run.sh"]
# CMD ["python3", "/app/run.py"]

#CMD /bin/bash
