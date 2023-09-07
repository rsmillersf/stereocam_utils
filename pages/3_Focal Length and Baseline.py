import streamlit as st
import cv2

st.set_page_config(layout="wide")

# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage("/apps/data/CamSetupParams.xml", cv2.FILE_STORAGE_READ)
focal = cv_file.getNode("focal").real()
baseline = cv_file.getNode("baseline").real()
pixelSize = cv_file.getNode("pixelSize").real()
cv_file.release()

st.title("Input Focal Length, Baseline Geometry, and Pixel Size")

focal = st.number_input('Camera focal length (in mm)', value=focal)

baseline = st.number_input("Stereo baseline (in cm)", value = baseline)

pixelSize = st.number_input("Camera horizontal pixel distance (in micrometers)", value = pixelSize)

save = st.button("Save camera setup params")
if save:
    cv_file = cv2.FileStorage("/apps/data/CamSetupParams.xml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("focal", focal)
    cv_file.write("baseline", baseline)
    cv_file.write("pixelSize", pixelSize)
    cv_file.release()
    st.write("Done!")