import numpy as np
import cv2

# def loadImage(self):
#     img = cv2.imread('./Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg')
#     cv2.imshow('Image', img)
#     print("Height = ", img.shape[0])
#     print("Width = ", img.shape[1])

# def colorsep(self):
#     img = cv2.imread('./Dataset_opencvdl/Q1_Image/Flower.jpg')
#     b, g, r = cv2.split(img)
#     zeros = np.zeros(b.shape, np.uint8)
#     blueBGR = cv2.merge((b,zeros,zeros))
#     greenBGR = cv2.merge((zeros,g,zeros))
#     redBGR = cv2.merge((zeros,zeros,r))
#     cv2.imshow('Blue', blueBGR)
#     cv2.imshow('Green', greenBGR)
#     cv2.imshow('Red', redBGR)

# def flipping(self):
#     img = cv2.imread('./Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg')
#     imgflip = cv2.flip(img, 1)
#     cv2.imshow('Result', imgflip)

# def blending(self):
#     cv2.namedWindow('Blending')
#     cv2.createTrackbar('BLEND: ', 'Blending', 0, 255, update_value)
#     img = cv2.imread('./Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg')
#     cv2.imshow('Blending', cv2.flip(img, 1))

# def update_value(x):
#     img = cv2.imread('./Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg')
#     imgflip = cv2.flip(img, 1)
#     dst = cv2.addWeighted(img, x/255, imgflip, 1 - x/255, 0.0)
#     cv2.imshow('Blending', dst)