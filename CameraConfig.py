import streamlit as st
import cv2
import numpy as np
import zmq
import utils

st.title("Stereo Camera Configuration and Rectification")

source = "/apps/data/left2.png"

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://127.0.0.1:5555")

#camera = cv2.VideoCapture("/dev/video0")
img = cv2.imread(source)
img = combs.scale(img)

counter = 0

while counter < 31:
    msg = socket.recv_string()
    frame = combs.spoof(img)
    md = combs.package(frame)
    socket.send_json(md, flags=zmq.SNDMORE)
    socket.send(frame, copy=True, track=False)
    counter += 1