#from RoiExtractorNew import ROIExtractorNew
#from CentroidTracker import CentroidTracker
import numpy as np
import cv2
import time
#import classifier
#import mainEngine
import fullTest

dev = True
classify = False
mainEngineRun = False
fullTester = True


def __init__():
    print("Initialised")


def main():
    """This is where the business happens"""
    """Load all our shit and get the main sectors allocated"""
    if fullTester:
        fullTest.run()
    if mainEngineRun:
        mainEngine.run()
        print("Dataset created")


    # classifier.classifyImage() MIIIIIIIIINT
# Top level code run
if __name__ == '__main__':
    __init__()
    main()
