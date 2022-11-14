FROM gitpod:workspace-python-tk

RUN sudo apt-get update && sudo apt-get install -y python3-opencv

RUN pip3 install pencv-python
RUN pip3 install opencv-contrib-python

RUN pip3 install mysql-connector
RUN pip3 install flask

RUN git clone https://github.com/carlinhoshk/Olhos_BublleDock.git

RUN cd Olhos_BublleDock/app

EXPOSE 80

CMD ["python3", "camera_flask_app.py"]