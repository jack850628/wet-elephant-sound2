from celery import Celery  
  
app = Celery('wet-elephant-sound2Task')
app.config_from_object('tasks.celeryconfig')
app.finalize()