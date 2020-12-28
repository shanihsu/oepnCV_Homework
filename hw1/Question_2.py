import numpy as np
import cv2

def medianfilter(self):
    img = cv2.imread('./Dataset_opencvdl/Q2_Image/Cat.png')
    cv2.imshow('median', cv2.medianBlur(img, 7))

def gaussianBlur(self):
    img = cv2.imread('./Dataset_opencvdl/Q2_Image/Cat.png')
    cv2.imshow('Gaussian', cv2.GaussianBlur(img, (3,3), 0))

def bilateralFilter(self):
    img = cv2.imread('./Dataset_opencvdl/Q2_Image/Cat.png')
    cv2.imshow('Bilateral', cv2.bilateralFilter(img,9,90,90))