import streamlit as st
import zmq
from utils import imgtools, mtxtools
import time
import cv2
import numpy as np
import zmq

st.set_page_config(layout="wide")

# Inner dimensons of chessboard, can change but
# results are better if don't include edge rows/columns
CHESS_X = 4
CHESS_Y = 6

# Find NUM corners before calc'ing intrinsic matrices
NUM = 10
found = 0

# Pull an image every WAIT_TIME seconds
WAIT_TIME = 1

# Source for video/image
IMG_SOURCE = "/apps/data/IMG_0416.JPG"
SOCKET = "tcp://127.0.0.1:5555"

# Page heading and column structure
st.title("Calibration")
run = st.checkbox("Run")

col1, col2 = st.columns(2)
with col1:
	FW1 = st.image([])
with col2:
	FW2 = st.image([])

##################################################################
# Spoof setting for testing, comment out for ZMQ run
img = cv2.imread(IMG_SOURCE)
img = imgtools.scale(img, "portrait")

# ZMQ settings, comment out for spoof testing
# context = zmq.Context()
# sock = context.socket(zmq.REC)
# sock.connect(SOCKET)
###################################################################

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
    ###############################################################
    # Grab combined image for spoof testing, comment of for production
    frame = imgtools.spoof(img)

    # Grab combined images for production, comment out for spoof testing
    # md = sock.recv_json()
    # msg = sock.recv(copy=True, track=False)
	# frame = np.frombuffer(msg, dtype=md["dtype"]).reshape(md["shape"])
    ################################################################

    # Split out L and R frames, resize to original, generate grayscale working copies
    imgL, imgR = imgtools.split(frame)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    if found < NUM:
        # Find the chess board corners
        retL, cornersL = cv2.findChessboardCornersSB(grayL, (CHESS_X,CHESS_Y),None)
        retR, cornersR = cv2.findChessboardCornersSB(grayR, (CHESS_X,CHESS_Y),None)

        # If found, add object points, image points after refining them
        if retL and retR:
            objpoints.append(objp)   # Certainly, every loop objp is the same, in 3D.
            corners2L = cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)
            corners2R = cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)
            imgpointsL.append(corners2L)
            imgpointsR.append(corners2R)

            # Draw and display the corners, increment found counter
            imgL = cv2.drawChessboardCorners(imgL, (CHESS_X,CHESS_Y), corners2L, retL)
            imgR = cv2.drawChessboardCorners(imgR, (CHESS_X,CHESS_Y), corners2R, retR)
            found += 1

    if found >= NUM:
        # We have enough, calc stero maps and save for disparity map cals
        st.write("Calculating stereo maps...")
        Left_Stereo_Map, Right_Stereo_Map = mtxtools.get_stereo_map(objpoints, imgpointsL, imgpointsR, grayL, grayR)

        st.write("Saving disparity maps...")
        cv_file = cv2.FileStorage("/apps/data/StereoMaps.xml", cv2.FILE_STORAGE_WRITE)
        cv_file.write("Left_Stereo_Map_x",Left_Stereo_Map[0])
        cv_file.write("Left_Stereo_Map_y",Left_Stereo_Map[1])
        cv_file.write("Right_Stereo_Map_x",Right_Stereo_Map[0])
        cv_file.write("Right_Stereo_Map_y",Right_Stereo_Map[1])
        cv_file.release()
        st.write("Done!")
        break

    # Display images in UI and wait before looping
    FW1.image(imgL)
    FW2.image(imgR)
    time.sleep(WAIT_TIME)
else:
    st.write("Stopped")
