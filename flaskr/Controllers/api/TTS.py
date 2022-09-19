import traceback
import flaskr.Model.TTS as tts
import flaskr.Utility.ResponseTemplate as ResponseTemplate
from flaskr.Config import Config
from flask import Blueprint, Response, request
from flaskr.Model.JException import *

TTS = Blueprint('TTS', __name__)

def init_app(app):
    app.register_blueprint(TTS, url_prefix='/api/TTS')

@TTS.route('/create_wav', methods=['POST'])
def createWav():
    try:
        text = request.values['text'].strip() if 'text' in request.values else '預設文字。'
        if text[-1] != '。':
            text += '。'
        return ResponseTemplate.success(tts.createWav(text))
    except (
        Exception
    ) as e:
        print(traceback.format_exc())
        response = ResponseTemplate.fail(e.__str__())
        response.status_code = 490
        return response