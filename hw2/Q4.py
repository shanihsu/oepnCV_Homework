import numpy as np
import cv2
from matplotlib import pyplot as plt

def computeDisparity():
    imgL = cv2.imread('./Datasets/Q4_Image/imgL.png',0)
    imgR = cv2.imread('./Datasets/Q4_Image/imgR.png',0)
    stereo = cv2.StereoBM_create(numDisparities=272, blockSize=11)
    disparity = stereo.compute(imgL,imgR)
    scale_percent =  40# percent of original size
    width = int(imgL.shape[1] * scale_percent / 100)
    height = int(imgL.shape[0] * scale_percent / 100)
    dim = (width, height)
    disparity = cv2.resize(disparity, dim)
    disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow('disparity', disparity)
    cv2.waitKey(0)
    # plt.imshow(disparity,'gray')
    # plt.show()
    
computeDisparity()