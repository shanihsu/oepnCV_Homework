import numpy as np
import cv2

def drawContour():
    img1 = cv2.imread('./Datasets/Q1_Image/coin01.jpg')
    img2 = cv2.imread('./Datasets/Q1_Image/coin02.jpg')
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.GaussianBlur(gray1,(5,5),0)
    blur2 = cv2.GaussianBlur(gray2,(5,5),0)
    low_threshold = 30
    high_threshold = 200
    ret, thresh1 = cv2.threshold(blur1, 127, 255, 0)
    ret, thresh2 = cv2.threshold(blur2, 127, 255, 0)
    edges1 = cv2.Canny(thresh1, low_threshold, high_threshold)
    edges2 = cv2.Canny(thresh2, low_threshold, high_threshold)
    contours1, hierarchy1 = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, hierarchy2 = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img1, contours1, -1, (0,0,255), 2)
    cv2.drawContours(img2, contours2, -1, (0,0,255), 2)
    cv2.imshow('coin01', img1)
    cv2.imshow('coin02', img2)

def countcoin1():
    img1 = cv2.imread('./Datasets/Q1_Image/coin01.jpg')
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.GaussianBlur(gray1,(5,5),0)
    low_threshold = 30
    high_threshold = 200
    ret, thresh1 = cv2.threshold(blur1, 127, 255, 0)
    edges1 = cv2.Canny(thresh1, low_threshold, high_threshold)
    contours1, hierarchy1 = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text = "There are  " +  str(len(contours1)) +  " coins in coin01.jpg"
    return text

def countcoin2():
    img2 = cv2.imread('./Datasets/Q1_Image/coin02.jpg')
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    blur2 = cv2.GaussianBlur(gray2,(5,5),0)
    low_threshold = 30
    high_threshold = 200
    ret, thresh2 = cv2.threshold(blur2, 127, 255, 0)
    edges2 = cv2.Canny(thresh2, low_threshold, high_threshold)
    contours2, hierarchy2 = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text = "There are  " +  str(len(contours2)) +  " coins in coin02.jpg"
    return text
