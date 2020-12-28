import numpy as np
import cv2
from matplotlib import pyplot as plt

def computeDisparity():
    """
    scale_percent =  40# percent of original size
    imgL = cv2.imread('./Datasets/Q4_Image/imgL.png',0)
    imgR = cv2.imread('./Datasets/Q4_Image/imgR.png',0)
    width = int(imgL.shape[1] * scale_percent / 100)
    height = int(imgL.shape[0] * scale_percent / 100)
    dim = (width, height)
    resizedL = cv2.resize(imgL, dim)
    resizedR = cv2.resize(imgR, dim)
    imgGrayL = cv2.cvtColor(resizedL, cv2.COLOR_BGR2GRAY)
    imgGrayR = cv2.cvtColor(resizedR, cv2.COLOR_BGR2GRAY)
    imtGrayL = cv2.equalizeHist(imgGrayL)
    imtGrayR = cv2.equalizeHist(imgGrayR)
    imgGrayL = cv2.GaussianBlur(imgGrayL, (5, 5), 0)
    imgGrayR = cv2.GaussianBlur(imgGrayR, (5, 5), 0)
    stereo = cv2.StereoBM_create(numDisparities=128, blockSize=11)
    disparity = stereo.compute(imgGrayL,imgGrayR).astype(np.float32)/16
    cv2.imshow('disparity', disparity)
    cv2.waitKey(0)
    
    
    imgL = cv2.imread('./Datasets/Q4_Image/imgL.png',0)
    imgR = cv2.imread('./Datasets/Q4_Image/imgR.png',0)
    stereo = cv2.StereoBM_create(numDisparities=272, blockSize=11)
    disparity = stereo.compute(imgL,imgR)
    plt.imshow(disparity,'gray')
    plt.show()
    """

    imgL = cv2.imread('./Datasets/Q4_Image/imgL.png',0)
    imgR = cv2.imread('./Datasets/Q4_Image/imgR.png',0)
    stereo = cv2.StereoBM_create(numDisparities=272, blockSize=11)
    disparity = stereo.compute(imgL,imgR).astype(np.float32)/16
    scale_percent =  40# percent of original size
    width = int(imgL.shape[1] * scale_percent / 100)
    height = int(imgL.shape[0] * scale_percent / 100)
    dim = (width, height)
    disparity = cv2.resize(disparity, dim)
    disparity = cv2.cvtColor(disparity, cv2.COLOR_GRAY2BGR)
    cv2.imshow('disparity', disparity)
    cv2.waitKey(0)

computeDisparity()