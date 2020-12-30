import numpy as np
import cv2
import glob

def findcorner():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((8*11,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('./Datasets/Q2_Image/*.bmp')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (11,8), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (11,8), corners2, ret)
            cv2.imshow('img', cv2.resize(img, (int(0.3*img.shape[1]), int(0.3*img.shape[0])), interpolation=cv2.INTER_CUBIC))
            cv2.waitKey(1500)
    cv2.destroyAllWindows()

def calibrateCamerafun(path):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((8*11,3), np.float32)
    np.set_printoptions(suppress=True)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob(path)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (11,8), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
    ret, intrinsicMtx, distortionMtx, _rvecs, _tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return intrinsicMtx, distortionMtx

def IntrinsicMatrix():
    intrinsicMtx, _ = calibrateCamerafun('./Datasets/Q2_Image/*.bmp')
    print(intrinsicMtx)

def DistortionMatrix():
    _, distortionMtx = calibrateCamerafun('./Datasets/Q2_Image/*.bmp')
    print(distortionMtx)

def ExtrinsicMatrix(imgindex):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((8*11,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    img = cv2.imread('./Datasets/Q2_Image/' + str(imgindex) + '.bmp')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (11,8), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        mtx, dist = calibrateCamerafun('./Datasets/Q2_Image/*.bmp')
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
        rotM = cv2.Rodrigues(rvecs)[0]
        print(np.concatenate((rotM, tvecs), axis=1))
    