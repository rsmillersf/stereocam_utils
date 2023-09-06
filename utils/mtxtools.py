import cv2
import numpy as np

#Calc intrinsics for a camera, expects grayscale image
def get_intrinsics(objpoints, imgpoints, frame):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame.shape[::-1], None, None)
    height, width = frame.shape
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
    return new_mtx, dist

#Use grayscale image for frame
def get_stereo_map(objpoints, imgpointsL, imgpointsR, frameL, frameR):
    new_mtxL, distL = get_intrinsics(objpoints, imgpointsL, frameL)
    new_mtxR, distR = get_intrinsics(objpoints, imgpointsR, frameR)
    
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC

    # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
    # Hence intrinsic parameters are the same 
    
    criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
    retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, new_mtxL, distL, new_mtxR, distR, frameL.shape[::-1], criteria_stereo, flags)

    # Perform stereo recification
    rectify_scale= 1
    rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR, frameL.shape[::-1], Rot, Trns, rectify_scale,(0,0))


    # Mapping equired to obtain the undistorted rectified stereo image pair
    Left_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,frameL.shape[::-1], cv2.CV_16SC2)
    Right_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,frameR.shape[::-1], cv2.CV_16SC2)

    return Left_Stereo_Map, Right_Stereo_Map