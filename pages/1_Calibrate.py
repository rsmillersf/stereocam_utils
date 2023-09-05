import streamlit as st
import zmq
from utils import imgtools
import time
import cv2
import numpy as np

st.title("Calibration")
run = st.checkbox("Run")

col1, col2 = st.columns(2)
with col1:
	FW1 = st.image([])
with col2:
	FW2 = st.image([])

source = "/apps/data/IMG_0416.JPG"
img = cv2.imread(source)
img = imgtools.scale(img, "portrait")

CHESS_X = 4
CHESS_Y = 6

NUM = 10
found = 0

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHESS_X*CHESS_Y,3), np.float32)
objp[:,:2] = np.mgrid[0:CHESS_X,0:CHESS_Y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane
imgpointsR = [] # 2d points in image plane

while run:
    frame = imgtools.spoof(img)
    imgL, imgR = imgtools.split(frame)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    if found < NUM:
        # Find the chess board corners
        retL, cornersL = cv2.findChessboardCornersSB(grayL, (CHESS_X,CHESS_Y),None)
        retR, cornersR = cv2.findChessboardCornersSB(grayR, (CHESS_X,CHESS_Y),None)

        # If found, add object points, image points (after refining them)
        if retL and retR:
            objpoints.append(objp)   # Certainly, every loop objp is the same, in 3D.
            corners2L = cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)
            corners2R = cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)
            imgpointsL.append(corners2L)
            imgpointsR.append(corners2R)

            # Draw and display the corners
            imgL = cv2.drawChessboardCorners(imgL, (CHESS_X,CHESS_Y), corners2L, retL)
            imgR = cv2.drawChessboardCorners(imgR, (CHESS_X,CHESS_Y), corners2R, retR)
            found += 1

    FW1.image(imgL)
    FW2.image(imgR)
    time.sleep(1)
else:
    st.write("Stopped")
