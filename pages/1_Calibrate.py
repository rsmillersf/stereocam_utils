import streamlit as st

st.title("Calibration")
run = st.checkbox("Run")

col1, col2 = st.columns(2)
with col1:
	FW1 = st.image([])
with col2:
	FW2 = st.image([])

#camera1 = cv2.VideoCapture("/dev/video0")
#camera2 = cv2.VideoCapture("/dev/video0")

while run:
	_, frame1 = camera1.read()
	_, frame2 = camera1.read()
	frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
	frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
	FW1.image(frame1)
	FW2.image(frame2)
else:
	st.write("Stopped")