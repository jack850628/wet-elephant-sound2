from flaskr import create_app
"""
uWsgi需要可以export的app物件，少了app將會啟動不了並報錯
Flask and uWSGI - unable to load app 0 (mountpoint='') (callable not found or import error)

"""
app = create_app()

if __name__ == "__main__":#由uWsgi啟動時_name_會等於uwsgi__file_app_App
    app.run(host = '0.0.0.0')