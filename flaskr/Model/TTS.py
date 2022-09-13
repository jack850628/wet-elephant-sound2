import uuid, os
import soundfile as sf
from flaskr.Model.JException import *
from flaskr.Values import Locales as Locales
from flaskr.Model.JException import *
from flaskr.TTS.test_all import production
from flaskr.TTS.util import hparams

output_dir = './flaskr/static/wav/'

def createWav(text):
    name = '{}.wav'.format(uuid.uuid1())
    sf.write(os.path.join(output_dir, name), production(text), hparams.sample_rate)
    return 'static/wav/{}'.format(name)
    