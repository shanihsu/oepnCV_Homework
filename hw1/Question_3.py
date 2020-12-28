import numpy as np
import cv2
import math

def gaussianBlur(self):
    img = cv2.imread('./Dataset_opencvdl/Q3_Image/Chihiro.jpg')
    x, y = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
    kernel = np.exp(-(np.square(x)+np.square(y))/(2*0.5)) # sigma = 0.707, sigma^2 = 0.5
    ginit = kernel/kernel.sum()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian = np.zeros(gray.shape, np.uint8)
    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            gaussian[i][j] = (gray[i-1:i+2, j-1:j+2] * ginit).sum()
    cv2.imshow('Gaussian', gaussian)

def sobelx(self):
    img = cv2.imread('./Dataset_opencvdl/Q3_Image/Chihiro.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g= [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel = np.zeros(gray.shape, np.uint8)
    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            sobel[i][j] = np.abs((gray[i-1:i+2, j-1:j+2] * g).sum())
    sobel = sobel/sobel.max()
    cv2.imshow('SobelX', sobel)

def sobely(self):
    img = cv2.imread('./Dataset_opencvdl/Q3_Image/Chihiro.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g= [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    sobel = np.zeros(gray.shape, np.uint8)
    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            sobel[i][j] = np.abs((gray[i-1:i+2, j-1:j+2] * g).sum())
    sobel = sobel/sobel.max()
    cv2.imshow('SobelY', sobel)

def magnitude(self):
    img = cv2.imread('./Dataset_opencvdl/Q3_Image/Chihiro.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx= [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    gy= [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    sobel = np.zeros(gray.shape, np.uint8)
    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            sobel_x = (gray[i-1:i+2, j-1:j+2] * gx).sum()
            sobel_y = (gray[i-1:i+2, j-1:j+2] * gy).sum()
            sobel[i][j] = np.abs(np.sqrt(np.square(sobel_x) + np.square(sobel_y)))
    sobel = sobel/sobel.max()
    cv2.imshow('Magnitude', sobel)