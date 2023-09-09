FROM ubuntu:latest

RUN	apt update && apt -y upgrade && apt install python-is-python3 python3-pip libgl1-mesa-dev libglib2.0-0  git -y

RUN pip install opencv-python streamlit pyzmq

WORKDIR /apps

RUN git clone https://github.com/rsmillersf/stereocam_utils.git

EXPOSE 5555
EXPOSE 8501

CMD streamlit run CameraConfig.py