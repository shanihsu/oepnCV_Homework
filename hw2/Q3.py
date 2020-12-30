import numpy as np
import cv2
import glob
import Q2

def AugmentedReality ():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((8*11,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
    axis = np.float32([[3,3,-3], [1,1,0], [3,5,0], [5,1,0]]).reshape(-1,3)
    mtx, dist = Q2.calibrateCamerafun('./Datasets/Q3_Image/*.bmp')
    for fname in glob.glob('./Datasets/Q3_Image/*.bmp'):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (11,8),None)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            # Find the rotation and translation vectors.
            ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            img = draw(img,corners2,imgpts)
            cv2.imshow('img',cv2.resize(img, (int(0.3*img.shape[1]), int(0.3*img.shape[0])), interpolation=cv2.INTER_CUBIC))
            cv2.waitKey(500)
    cv2.destroyAllWindows()

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0,0,255), 5)
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0,0,255), 5)
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 5)
    img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()), (0,0,255), 5)
    img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 5)
    img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 5)
    return img
