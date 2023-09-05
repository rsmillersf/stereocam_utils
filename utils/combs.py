import cv2
import numpy as np

#Spoofs a combination of two camera images, needs error checking
# assumes landscape orientation
def spoof(frame, tiles=2, interpolation="INTER_AREA"):
    interpolation = "cv2." + interpolation
    height, width, _ = frame.shape
    dim = (int(width/tiles), height) # h and w are reversed re: shape
    img = cv2.resize(frame, dim, interpolation = interpolation)
    return cv2.hconcat([img, img])

#Splits an image into two streams and resizes each up
def split(frame, tiles=2, interpolation = "INTER_AREA"):
    interpolation = "cv2." + interpolation
    height, width, _ = frame.shape
    dim = (width, height) # h and w are reversed re: shape
    midpoint = int(width/tiles)
    imgL = cv2.resize(frame[:,:midpoint,:], dim, interpolation = cv2.INTER_LINEAR)
    imgR = cv2.resize(frame[:,midpoint:,:], dim, interpolation = cv2.INTER_LINEAR)
    return imgL, imgR