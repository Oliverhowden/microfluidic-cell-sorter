import cv2
import numpy as np

"""ROI Extractor class"""
"""Blur, Dilate, Erode, and perform Bitwise Operations """
"""This class takes a square ROI and responds with the actual cell present in that image"""


class ROIExtractorNew():

    def init(self):
        self.devMode = False

    def extractROI(self, roi, mask):
        gray_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        height, width = gray_img.shape
        blur_kernal = ((5, 5), 1)
        erode_kernal = (5, 5)
        dilate_kernal = (3, 3)
        kernel = np.ones((5, 5), np.uint8)
        fullMask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel).astype(np.uint8)
        edges = cv2.Canny(fullMask, 100, 200)
        res1 = cv2.bitwise_and(roi, roi, mask=fullMask)

        if self.devMode:
            cv2.imshow('Pre White Mask', fullMask)
        # Set all kernals for processing methods
        # Touchup edges of mask

        fullMask = cv2.dilate(cv2.erode(cv2.GaussianBlur(
            fullMask, blur_kernal[0], blur_kernal[1]), np.ones(erode_kernal)), np.ones(dilate_kernal))

        # Response from bitwise mask should be final cell only
        res = cv2.bitwise_and(roi, roi, mask=fullMask)

        # TODO Maybe we want to run these to the GUI window, in which case these will need to have a better method of exporting themselves
        if self.devMode:
            cv2.imshow('Initial Mask', gray_img)
            cv2.imwrite('InitialMask.png', gray_img)

        if self.devMode:
            cv2.imshow('recievedMask', mask)
            cv2.imwrite('RecievedMask.png', mask)
        if self.devMode:
            cv2.imshow('White Mask', fullMask)
            cv2.imwrite('WhiteMask.png', fullMask)
        if self.devMode:
            cv2.imshow('Canny Edge Detected', edges)
            cv2.imwrite('EdgeDetected.png', edges)
        if self.devMode:
            cv2.imshow('Extracted full', res)
            cv2.imwrite('Final Extraction.png', res)
        if self.devMode:
            cv2.imshow('Extracted pre', res1)
            cv2.imwrite('Initial Masked.png', res1)
            # input('enter')
        cv2.imshow('insideSet', res)
        return res
