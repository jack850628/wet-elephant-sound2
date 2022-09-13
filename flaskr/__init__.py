import sys
sys.path.append("./flaskr/TTS")

import os
from flask import Flask
from flaskr.Controllers.api import TTS
from flaskr.Config import Config
from flask import render_template

def create_app():
    app = Flask(__name__)
    app.debug = Config.DEBUG
    app.config['TESTING'] = Config.TESTING
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')

    TTS.init_app(app)


    @app.route("/", methods=['GET'])
    def index():
        # from tasks import CreateThumbnail as task
        # task.hello.delay("安安")
        return render_template("index.html")

    return app