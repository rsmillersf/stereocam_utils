import streamlit as st
import zmq
from utils import combs
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
img = combs.scale(img, "portrait")

while run:
    frame = combs.spoof(img)
    imgL, imgR = combs.split(frame)
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    FW1.image(imgL)
    FW2.image(imgR)
    time.sleep(1)
else:
    st.write("Stopped")

'''

#camera1 = cv2.VideoCapture("/dev/video0")
#camera2 = cv2.VideoCapture("/dev/video0")

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:5555")

while run:
    socket.send_string("ready")
    md = socket.recv_json()
    frame = socket.recv(copy=True, track=False)
    frame = np.frombuffer(msg, dtype=md["dtype"]).reshape(md["shape"])
    imgL, imgR = combs.split(frame)
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    FW1.image(imgL)
    FW2.image(imgR)
    time.sleep(1)
else:
	st.write("Stopped")
'''

