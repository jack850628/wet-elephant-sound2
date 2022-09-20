BROKER_URL = 'redis://127.0.0.1:6379'
CELERY_RESULT_BACKEND = 'redis://127.0.0.1:6379'

CELERY_TIMEZONE='Asia/Taipei'                     # 設定時區，默认是預設是UTC
  
CELERY_IMPORTS = (                                  # 匯入task 
    'tasks.TTS',
)  