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
        userID = request.values['userID'].strip() if 'userID' in request.values else None
        text = request.values['text'].strip() if 'text' in request.values else '預設文字。'
        if userID is None:
            raise '缺少userID。'
        if text[-1] != '。':
            text += '。'
        tts.createWav(userID, text)
        return ResponseTemplate.success('OK')
    except (
        Exception
    ) as e:
        print(traceback.format_exc())
        response = ResponseTemplate.fail(e.__str__())
        response.status_code = 490
        return response

@TTS.route('/wait_audio_name', methods=['GET'])
def waitAudioName():
    try:
        userID = request.values['userID'].strip() if 'userID' in request.values else None
        if userID is None:
            raise '缺少userID。'
        return Response(tts.waitAudioName(userID), content_type='text/event-stream')
    except (
        Exception
    ) as e:
        print(traceback.format_exc())
        response = ResponseTemplate.fail(e.__str__())
        response.status_code = 490
        return response