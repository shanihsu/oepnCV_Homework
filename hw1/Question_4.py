import numpy as np
import cv2

def to_num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def tsf(angle, scale, tx, ty):
    img = cv2.imread('./Dataset_opencvdl/Q4_Image/Parrot.png')
    angle = to_num(angle) if angle else 0
    scale = to_num(scale) if scale else 0
    tx = to_num(tx) if tx else 0
    ty = to_num(ty) if ty else 0
    height,width = img.shape[:2]

    #translation
    H = np.float32([[1, 0, tx], [0, 1, ty]])
    rst = cv2.warpAffine(img, H, (width, height))

    #rotate and scale
    center = (160+tx, 84+ty)
    if scale == 0:
        scale = 1
    M = cv2.getRotationMatrix2D(center, angle, float(scale))
    rst = cv2.warpAffine(rst, M, (width, height))
    
    cv2.imshow('Original Image', img)
    cv2.imshow('Image RST', rst)
