import os, glob
import pathlib
import JTools as JTools
import soundfile as sf
import redis

from tasks import app
from flaskr.TTS.test_all import production
from flaskr.TTS.util import hparams

path = str(pathlib.Path(__file__).parent.parent.absolute()) + '/flaskr/static/wav'
pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
r = redis.Redis(connection_pool=pool)

@app.task
def doTTS(userID: str, fileName: str, text: str):
    sf.write(os.path.join(path, fileName), production(text), hparams.sample_rate)
    if len(r.hgetall(userID + '佔位')) != 0:
        r.rpush(userID, fileName)
