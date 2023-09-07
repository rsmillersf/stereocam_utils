FROM ubuntu:latest

RUN	apt update && apt -y upgrade && apt install python-is-python3 python3-pip libgl1-mesa-dev libglib2.0-0  git -y

RUN pip install opencv-python streamlit pyzmq