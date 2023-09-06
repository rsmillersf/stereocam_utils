import streamlit as st
import cv2
from utils import imgtools
import numpy as np

st.title("Set Disparity Map Parameters")

source = "/apps/data/IMG_0416.JPG"
img = cv2.imread(source)
img = imgtools.scale(img, "portrait")
stereo = cv2.StereoBM_create()

st.write("Loading params")

# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage("/apps/data/improved_params2.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()
st.write("Done!")

col1, col2 = st.columns(2)

with col1:
    numDisparities = st.slider("Number of Disparities", 1, 17)*16
    blockSize = st.slider("Block Size", 5, 50)*2 + 5
    preFilterType = st.select_slider("Pre-Filter Type", options = ["CV_STEREO_BM_XSOBEL", "CV_STEREO_BM_NORMALIZED_RESPONSE"])
    preFilterSize = st.slider("Pre-Filter Size", 2, 25)*2 + 5
    preFilterCap = st.slider("Pre-Filter Cap", 5, 62)
    textureThreshold = st.slider("Texture Threshold", 10, 100)
    uniquenessRatio = st.slider("Uniqueness Ratio", 15, 100)
    speckleRange = st.slider("Speckle Range", 0, 100)
    speckleWindowSize = st.slider("Speckle Window Size", 3, 25)*2
    disp12MaxDiff = st.slider("Disp to Max Diff", 5, 25)
    minDisparity = st.slider("Min Disparity", 5, 25)

with col2:
    run = st.checkbox("Run")
    frame_window = st.image([])

while run:
    frame = imgtools.spoof(img)
    imgL, imgR = imgtools.split(frame)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Applying stereo image rectification on the left image
    Left_nice = cv2.remap(grayL, 
        Left_Stereo_Map_x, 
        Left_Stereo_Map_y, 
        cv2.INTER_LANCZOS4,
        cv2.BORDER_CONSTANT,
        0)
     
    # Applying stereo image rectification on the right image
    Right_nice= cv2.remap(grayR,
        Right_Stereo_Map_x,
        Right_Stereo_Map_y,
        cv2.INTER_LANCZOS4,
        cv2.BORDER_CONSTANT,
        0)

    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterType(1)
    stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)
 
    # Calculating disparity using the StereoBM algorithm
    disparity = stereo.compute(Left_nice,Right_nice)
    # NOTE: Code returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it 
    # is essential to convert it to CV_32F and scale it down 16 times.
 
    # Converting to float32 
    disparity = disparity.astype(np.float32)
 
    # Scaling down the disparity values and normalizing them 
    disparity = (disparity/16.0 - minDisparity)/numDisparities
 
    # Displaying the disparity map
    frame_window.image(disparity, clamp = True)
 
    # Close window using esc key
    if cv2.waitKey(1) == 27:
      break