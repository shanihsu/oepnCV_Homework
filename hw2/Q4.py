import numpy as np
import cv2
from matplotlib import pyplot as plt

class disparityimg:
    def __init__(self):
        self.disparity = None
        self.previousimg = None
        cv2.namedWindow("disparity", cv2.WINDOW_NORMAL)
        self.text1 = ""
        self.text2 = ""

    def computeDisparity(self):
        imgL = cv2.imread('./Datasets/Q4_Image/imgL.png',0)
        imgR = cv2.imread('./Datasets/Q4_Image/imgR.png',0)
        stereo = cv2.StereoBM_create(numDisparities=272, blockSize=11)
        self.disparity = stereo.compute(imgL,imgR)
        self.previousimg = stereo.compute(imgL,imgR)
        print(type(self.disparity))
        print(self.disparity)
        print(self.disparity.min())
        print(self.disparity.max())
        #scale_percent =  40# percent of original size
        #width = int(imgL.shape[1] * scale_percent / 100)
        #height = int(imgL.shape[0] * scale_percent / 100)
        #dim = (width, height)
        #self.disparity = cv2.resize(self.disparity, dim)
        #self.previousimg = cv2.resize(self.disparity, dim)
        self.disparity = cv2.normalize(self.disparity, self.disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.previousimg = cv2.normalize(self.previousimg, self.previousimg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        print(self.disparity)
        print(self.disparity.min())
        print(self.disparity.max())
        cv2.namedWindow("disparity", cv2.WINDOW_NORMAL)
        cv2.imshow('disparity', self.disparity)
        
        #cv2.waitKey(0)

    def onMouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            #self.computeDisparity()
            self.disparity = self.previousimg.copy()
            #cv2.imshow('disparity', self.previousimg)
            cv2.circle(self.disparity,(x,y),10,(255,0,0),-1)
            cv2.namedWindow("disparity", cv2.WINDOW_NORMAL)
            cv2.imshow('disparity', self.disparity)
            print(x , " ", y)
            print(self.disparity[x][y])

    def selectpoint(self):
        self.computeDisparity()
        x,y,w,h = self.disparity.shape[1],self.disparity.shape[0],700,300
        # Draw white background rectangle
        cv2.rectangle(self.disparity, (x-700, y-300), (x + h, y + w), (255,255,255), -1)
        self.text1 = "Disparity: " + str(self.disparity[0][0]) + " pixels"
        self.text2 = "Depth: " + str(self.calculatedepth(self.disparity[0][0])) + " mm"
        # Add text
        cv2.putText(self.disparity, self.text1, (x - 700 + int(w/10),y-300 + int(h/4)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2)
        cv2.putText(self.disparity, self.text2, (x - 700 + int(w/10),y-300 + int(h/4)+int(h/4)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2)
        cv2.namedWindow("disparity", cv2.WINDOW_NORMAL)
        cv2.imshow('disparity', self.disparity)
        cv2.setMouseCallback('disparity', self.onMouse)
        cv2.waitKey(0)

    def calculatedepth(self, disparityvalue):
        baseline = 178
        focallength = 2826 
        minus_cr_cl = 123
        depth = baseline * focallength / (minus_cr_cl + disparityvalue) 
        return int(depth)

test = disparityimg()
test.selectpoint()