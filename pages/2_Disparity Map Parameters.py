import streamlit as st
import cv2
from utils import imgtools
import numpy as np

st.set_page_config(layout="wide")

IMG_SOURCE = "/apps/data/IMG_0416.JPG"
SOCKET = "tcp://127.0.0.1:5555"

# Basic page layout, columns elements filled in after loading saved params
st.title("Set Disparity Map Parameters")
col1, col2, col3 = st.columns(3, gap="medium")

##################################################################
# Spoof setting for testing, comment out for ZMQ run
img = cv2.imread(IMG_SOURCE)
img = imgtools.scale(img, "portrait")

# ZMQ settings, comment out for spoof testing
# context = zmq.Context()
# sock = context.socket(zmq.REC)
# sock.connect(SOCKET)
###################################################################

# Create Block Matching object
stereo = cv2.StereoBM_create()

# Reading the saved mapping values for stereo image rectification
cv_file = cv2.FileStorage("/apps/data/StereoMaps.xml", cv2.FILE_STORAGE_READ)
if cv_file.isOpened():
    Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
    Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
    Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
    Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
    cv_file.release()
else:
    st.warning("Could not open Stero Maps. Please complete camera calibration step and try again.")
    st.stop()

# Reading the bock matching params
sbParams = 1
cv_file = cv2.FileStorage("/apps/data/BlockMatchingParams.xml", cv2.FILE_STORAGE_READ)
if cv_file.isOpened():
    numDisparities = cv_file.getNode("numDisparities").real()
    blockSize = int(cv_file.getNode("blockSize").real())
    #preFilterType = cv_file.getNode("preFilterType").real()
    preFilterSize = cv_file.getNode("preFilterSize").real()
    preFilterCap = cv_file.getNode("preFilterCap").real()
    textureThreshold = cv_file.getNode("textureThreshold").real()
    uniquenessRatio = cv_file.getNode("uniquenessRatio").real()
    speckleRange = cv_file.getNode("speckleRange").real()
    speckleWindowSize = cv_file.getNode("speckleWindowSize").real()
    disp12MaxDiff = cv_file.getNode("disp12MaxDiff").real()
    minDisparity = cv_file.getNode("minDisparity").real()
    cv_file.release()
else:
    sbParams = 0

# Block Matching params
with col1:
    #numDisparities = st.slider("Number of Disparities", 1*16, 17*16, int(numDisparities), 16)
    numDisparities = st.slider("Number of Disparities", 1*16, 17*16, **({"value":int(numDisparities)} if sbParams else {}), step=16)
    blockSize = st.slider("Block Size", (5*2+5), (50*2+5), **({"value":int(blockSize)} if sbParams else {}), step=2)
    #preFilterType = st.select_slider("Pre-Filter Type", options = ["CV_STEREO_BM_XSOBEL", "CV_STEREO_BM_NORMALIZED_RESPONSE"])
    preFilterSize = st.slider("Pre-Filter Size", 9, 55, **({"value":int(preFilterSize)} if sbParams else {}), step=2)
    preFilterCap = st.slider("Pre-Filter Cap", 5, 62, **({"value":int(preFilterCap)} if sbParams else {}))
    textureThreshold = st.slider("Texture Threshold", 10, 100, **({"value":int(textureThreshold)} if sbParams else {}))
    
# More Block Matching params
with col2:
    uniquenessRatio = st.slider("Uniqueness Ratio", 15, 100, **({"value":int(uniquenessRatio)} if sbParams else {}))
    speckleRange = st.slider("Speckle Range", 0, 100, **({"value":int(speckleRange)} if sbParams else {}))
    speckleWindowSize = st.slider("Speckle Window Size", 3*2, 25*2, **({"value":int(speckleWindowSize)} if sbParams else {}), step=2)
    disp12MaxDiff = st.slider("Disp to Max Diff", 5, 25, **({"value":int(disp12MaxDiff)} if sbParams else {}))
    minDisparity = st.slider("Min Disparity", 5, 25, **({"value":int(minDisparity)} if sbParams else {}))

# View disparity map, adjust as needed using param sliders
with col3:
    run = st.checkbox("Run")
    frame_window = st.image([])
    save = st.button("Save block matching params")

while run:
    ###############################################################
    # Grab combined image for spoof testing, comment of for production
    frame = imgtools.spoof(img)

    # Grab combined images for production, comment out for spoof testing
    # md = sock.recv_json()
    # msg = sock.recv(copy=True, track=False)
	# frame = np.frombuffer(msg, dtype=md["dtype"]).reshape(md["shape"])
    ################################################################
    
    # Split out L and R frames and resize, create grayscale working copies
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

    # Save params on button press
    if save:
        col2.write("Saving...")
        cv_file = cv2.FileStorage("/apps/data/BlockMatchingParams.xml", cv2.FILE_STORAGE_WRITE)
        cv_file.write("numDisparities", numDisparities)
        cv_file.write("blockSize", blockSize)
        #cv_file.write("preFilterType", preFilterType)
        cv_file.write("preFilterSize", preFilterSize)
        cv_file.write("preFilterCap", preFilterCap)
        cv_file.write("textureThreshold", textureThreshold)
        cv_file.write("uniquenessRatio", uniquenessRatio)
        cv_file.write("speckleRange", speckleRange)
        cv_file.write("speckleWindowSize", speckleWindowSize)
        cv_file.write("disp12MaxDiff", disp12MaxDiff)
        cv_file.write("minDisparity", minDisparity)
        cv_file.release()
        col3.write("Done!")
        save = 0

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