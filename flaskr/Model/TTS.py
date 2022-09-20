import uuid, time
import redis
from flaskr.Model.JException import *
from flaskr.Values import Locales as Locales
from flaskr.Model.JException import *
from tasks import TTS as tts

pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
r = redis.Redis(connection_pool=pool)

def createWav(userID, text):
    name = '{}.wav'.format(uuid.uuid1())
    tts.doTTS.delay(userID, name, text)
    # return 'static/wav/{}'.format(name)

def waitAudioName(userID: str):
    r.hset(userID + '佔位', '佔位', '佔位資料')
    try:
        while True:
            audioName = r.lpop(userID)
            if audioName != None:
                yield "event: audioName\ndata: %s\n\n" % audioName
            else:
                yield "event: heartbeat\ndata: ok\n\n"
            time.sleep(1)
    finally:
        r.delete(userID)
        r.delete(userID + '佔位')

    