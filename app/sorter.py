import cv2
import numpy as np
import os
import shutil
import datetime
import time
from collections import OrderedDict
from customclasses.RoiExtractorNew import ROIExtractorNew
from customclasses.Tracker import Tracker
from customclasses import classifier
from augmentDat import Augmentation
import serial

simulationData = True

if simulationData:
    # Use our recorded video from the acquisition
    video = cv2.VideoCapture("../data/8um-beads_20um-channel.avi")
else:
    # Connect to the first available video feed
    video = cv2.VideoCapture(0)


# functional variables
createDataset = False
ignoreClassifier = False

# global
dev = False
arduino = False

# sensitivity
minimumCertainty = 0.97

# bounding
bounds = 480
classificationBounds = 20
roiPadding = 20
frame_width = int(video.get(3))
frame_height = int(video.get(4))

# kernals
thresh_sensitivity = 20
bounding_pad = 2
kern = np.ones((3, 3), np.uint8)

# resulting display
resultText = 'Result'
resultColour = (0, 0, 0)

# various
evalCount = 0
evalTotal = 0.0


if createDataset:
    ignoreClassifier = True
else:
    ignoreClassifier = False

if arduino:
    ser = serial.Serial('/dev/tty.usbmodem14101', 9600)


# Create a dataset directory to store acquired images
dirname = 'created_dataset'

# Check if the directory exists
if os.path.isdir(dirname):

    # if its already there, remove it and recreate it
    shutil.rmtree(dirname, ignore_errors=True)
os.mkdir(dirname)

# Initialise middlePoint Tracker
ct = Tracker()
ct.init(bounds)

# Initialise ROI Extractor
extractor = ROIExtractorNew()
extractor.init()

# Augmentor
aug = Augmentation()

# Read an initial image from the video stream to use
# as noise reduction, also to ensure that the stream is ok
ok1, base_img = video.read()


def classifyROI(roi, cellID):
    global evalCount
    global evalTotal

    if not ignoreClassifier:
        eval_time, labels, score = classifier.classifyImage(roi)
        evalCount += 1
        evalTotal += eval_time
    else:
        score = 0
        labels = ''
        eval_time = 0.0
    print('\nEvaluation time (1-image): {:.3f}s\n'.format(eval_time))
    template = "{} (score={:0.5f})"
    print(template.format(labels, score))
    if score > minimumCertainty:
        ct.setClassificationResult(cellID, True)
    else:
        ct.setClassificationResult(cellID, False)

    return  # score


def processFramesReturnContours(frame1, frame2):
    difference_img = cv2.absdiff(frame1, frame2)
    if dev:
        cv2.imshow('Absolute Difference', difference_img)
        cv2.imwrite(os.path.join(
            dirname, '{}.png'.format('1AbsDiff')), difference_img)

    ret, thresholded_img = cv2.threshold(
        difference_img, thresh_sensitivity, 255, cv2.THRESH_BINARY)

    if dev:
        cv2.imshow('Initial Thesholded', thresholded_img)
        cv2.imwrite(os.path.join(
            dirname, '{}.png'.format('2InitThresh')), thresholded_img)

    thresholded_img = cv2.morphologyEx(
        thresholded_img, cv2.MORPH_CLOSE, kern)

    if dev:
        cv2.imshow('Morphological Close', thresholded_img)
        cv2.imwrite(os.path.join(
            dirname, '{}.png'.format('3MorphClose')), thresholded_img)

    thresholded_img = cv2.dilate(thresholded_img, kern, iterations=1)

    if dev:
        cv2.imshow('Dilated Image', thresholded_img)
        cv2.imwrite(os.path.join(
            dirname, '{}.png'.format('4Dilated')), thresholded_img)

    thresholded_img = cv2.GaussianBlur(thresholded_img, (3, 3), 1)

    if dev:
        cv2.imshow('Blurred Image', thresholded_img)
        cv2.imwrite(os.path.join(
            dirname, '{}.png'.format('5Blurred')), thresholded_img)

    ret, thresholded_img = cv2.threshold(
        thresholded_img, thresh_sensitivity, 255, cv2.THRESH_BINARY)

    if dev:
        cv2.imshow('Thresholded again', thresholded_img)
        cv2.imwrite(os.path.join(
            dirname, '{}.png'.format('6SecondThresh')), thresholded_img)

    im2, contours, hierarchy = cv2.findContours(
        thresholded_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if dev:
        cv2.imshow('Contours', im2)
        cv2.imwrite(os.path.join(
            dirname, '{}.png'.format('7Contours')), im2)
    frame1 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(frame1, contours, 0, (102, 255, 102), 1)
    # print(hierarchy)
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    if dev:
        cv2.imshow('Final output', frame1)
        cv2.imwrite(os.path.join(
            dirname, '{}.png'.format('8Final')), frame1)
        input("Press Enter to continue...")

    return contours, thresholded_img


def returnAllKnownCellDataForAllCells():
    allIds = ct.returnAllKnownCellIDs()
    allEverything = OrderedDict()
    for id in allIds:
        allEverything[id] = ct.returnAllCellResults(id)
    return allEverything


def getFrame():
    global resultText
    global resultColour

    # Read two frames.
    ok, frame1 = video.read()
    ok2, frame2 = video.read()

    # check video stream is valid
    if ok and ok2:
        # createa a reference frame
        unprocessed_reference_img = frame2.copy()

        # convert both frames to grey
        frame1 = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        contours, thresholded_img = processFramesReturnContours(frame1, frame2)
        rects = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # print(w, h)
            if w > 10:
                if cv2.contourArea(contour, True) < 0:
                    if (x + 2 * h > 10):
                        sX = int(x)
                        eX = int(x + w)
                        sY = int(y)
                        eY = int(y + h)
                        cv2.rectangle(frame2, (sX, sY),
                                      (eX, eY), (0, 0, 255), 1)

                        rect = np.array([sX, sY, eX, eY])
                        rects.append(rect.astype("int"))
                        # print(x + 2 * w)
                        # input('wait')
        objects = ct.updateTracker(rects)

        for (cellID, middlePoint) in objects.items():
            # draw both the ID of the object and the middlePoint of the
            # object on the output frame
            text = "ID {}".format(hex(cellID))
            cv2.putText(frame2, text, (middlePoint[0] - 10, middlePoint[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.circle(
                frame2, (middlePoint[0], middlePoint[1]), 2, (0, 255, 0), -1)

            # Set roiPadding from middlePoint
            # Extract Rough ROI
            roi = unprocessed_reference_img[middlePoint[1] - roiPadding:middlePoint[1] + roiPadding,
                                            middlePoint[0] - roiPadding: middlePoint[0] + roiPadding]

            mask = thresholded_img[middlePoint[1] - roiPadding:middlePoint[1] + roiPadding,
                                   middlePoint[0] - roiPadding: middlePoint[0] + roiPadding]

            # Check size > 0, if so threshold/edge detect/etc
            if (np.size(roi, 0) > 5) and (np.size(roi, 1) > 5):

                if ct.hasThisCellBeenClassified(cellID):

                    #print("Already classified")

                    if (ct.returnCurrentYFromCellID(cellID) > bounds) and (ct.getClassificationResult(cellID) == True):
                        if arduino:
                            ser.write(b'H')
                        print("Kept")
                        resultText = 'Kept = True'
                        resultColour = (150, 255, 100)

                    elif (ct.returnCurrentYFromCellID(cellID) > bounds) and (ct.getClassificationResult(cellID) == False):
                        if arduino:
                            ser.write(b'L')
                        resultText = 'Kept = False'
                        resultColour = (255, 0, 0)

                        print("Wasted")
                else:
                    if middlePoint[1] < 50:
                        finalROI = extractor.extractROI(roi, mask)

                        if dev:
                            cv2.imshow('SetROI', finalROI)
                        classifyROI(roi, cellID)
                        # ct.setClassificationBoolTrue(cellID)
                        ct.setRoiForID(finalROI, cellID)

                        print("Classifying")
                        if createDataset:
                            cv2.imwrite(os.path.join(
                                dirname, '{}.png'.format(cellID)), roi)

        frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2RGB)

        #input('Step to next cell')
        if not ignoreClassifier:
            cv2.putText(frame2, resultText, (400, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.circle(frame2, (390, 445), 6, resultColour, -1)

        cv2.imshow('cont', frame2)
        cv2.imshow('thresh', thresholded_img)
        cv2.imshow('original', unprocessed_reference_img)
        # input('Enter')
    return ok, frame2


def stop():
    print('stopping')
    global video
    video.release()


k = 0


def run():

    now = time.time()

    while True:
        global k
        # 25 fps
        time.sleep(0.05)
        ok, frame = getFrame()
        if ok:
            cv2.imshow('Final App', frame)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if not createDataset:
        print('totalsessionTime={}'.format(now - time.time()))
        print('Overalltime={}'.format(evalTotal / evalCount))
    if createDataset:
        aug.runAugmentation(os.path.join(dirname))


# When everything is done, release the capture
cv2.destroyAllWindows()
