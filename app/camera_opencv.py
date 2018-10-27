import cv2
import numpy as np
import datetime
import time
from collections import OrderedDict
from customclasses.RoiExtractorNew import ROIExtractorNew
from customclasses.Tracker import Tracker
#from customclasses import processor
import sys
import fullTest
import cv2
from base_camera import BaseCamera
import time
import json


class Camera(BaseCamera):
    video_source = 0

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture("../videos/8.avi")
        if not camera.isOpened():
            raise RuntimeError('Could not start video.')
        k = 0

        while True:
            # read current frame
            k = k + 1
            time.sleep(0.1)
            s, img = fullTest.getFrame()  # camera.read()
            cv2.imshow('mmm', img)
            # encode as a jpeg image and return it
            if s:
                yield cv2.imencode('.jpg', img)[1].tobytes()
            else:
                print("Stream Finished", file=sys.stderr)
                break
