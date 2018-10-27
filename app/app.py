#!/usr/bin/env python
from importlib import import_module
import os
from flask import Flask, render_template, Response
import fullTest
# import camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera_opencv import Camera

# Raspberry Pi camera module (requires picamera package)
# from camera_pi import Camera

app = Flask(__name__)


@app.route('/train.html')
def train():
    """Training home page."""
    return render_template('train.html')


@app.route('/createdataset.html')
def createdataset():
    """Training home page."""
    return render_template('createdataset.html')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. """
    # fullTest.run()
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop')
def stop():
    fullTest.stop()


@app.route('/start')
def start():
    """Video streaming route. """
    print('start')
    video_feed()


@app.route('/videofeed')
def videofeed():
    """Video streaming route"""
    # return Response(gen(Camera()),
    #    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
