from fileinput import filename
from tkinter import Frame, image_names
from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import mysql.connector
import os, sys
import numpy as np
from threading import Thread


global capture,rec_frame, grey, switch, neg, face, rec, out
capture=0
grey=0
neg=0
face=0
switch=1
rec=0
cadastro=0
loop=0

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#Load pretrained face detection model    
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

#instatiate flask app  
app = Flask(__name__, template_folder='./frontend/templates', static_folder="./frontend/static")

contador = 1


#-banco-#
#app_user #password
connection = mysql.connector.connect(user='banco-casa.mysql.database.azure.com', password='gtasandreas1:',
                            host='banco-casa.mysql.database.azure.com',
                            database='banco_casa')
mycurso = connection.cursor()
                       

sql_query = """INSERT INTO entradaESaida (contador_de_pessoa, dia_entrada, horario_entrada, semana, frame) VALUES (%s,%s,%s,%s,%s)"""

font = cv2.FONT_HERSHEY_DUPLEX
font1 = cv2.FONT_HERSHEY_DUPLEX
months = ['janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho', 'julho',
'Agosto', 'setembro', 'outubro', 'novembro', 'dezembro']
weekdays = ['Segunda', 'terça', 'quarta', 'quinta', 'sexta', 'sábado', 'domingo']


camera = cv2.VideoCapture("ibmface.mp4")

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)

def salva_foto_banco(frame):
    global byte_encode

    img_encode = cv2.imencode('.jpg', frame)[1]
    data_encode = np.array(img_encode)

    byte_encode = data_encode.tobytes()

    return byte_encode

def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:            
            return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        pass
    return frame
 

def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame, loop, contador
    while True:
        success, frame = camera.read() 
        if success:
            if(face):                
                frame= detect_face(frame)
                if(loop==1):
                    loop=0
                    now = datetime.datetime.now()
                    curTime=time.time()
                    localTime = time.localtime(curTime)
                    horario = "%dh:%dm:%ds" % (localTime.tm_hour,localTime.tm_min,localTime.tm_sec)
                    data = localTime.tm_mday, months[localTime.tm_mon-1], localTime.tm_year
                    semana = weekdays[localTime.tm_wday]
                    contador+=1

                    
                    bite_frame = salva_foto_banco(frame)
                    getbanco = (contador, str(data), horario, semana, bite_frame)     
                    mycurso.execute(sql_query, getbanco)
                    connection.commit()
                    print("Dados subiu para o banco")
                    p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                    cv2.imwrite(p, frame)
                    #print(frame)
            if(grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if(neg):
                frame=cv2.bitwise_not(frame)    
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
            
            if(rec):
                rec_frame=frame
                frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                frame=cv2.flip(frame,1)
            
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass


def falaoi(x):
    a = x+2

    return a

@app.route('/')
def index():
    return render_template('index.html', image_names=filename )
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('grey') == 'Grey':
            global grey
            grey=not grey
        elif  request.form.get('neg') == 'Negative':
            global neg
            neg=not neg

        elif  request.form.get('cad') == 'Cadastro':
            global loop

            loop=1
           
        elif  request.form.get('face') == 'Face Only':
            global face

            face=not face 
            
            if(face):
                time.sleep(4)

        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
        elif  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()
                          
                 
    elif request.method=='GET':
        return render_template('index.html', image_names=filename)
    return render_template('index.html', image_names=filename)


if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()     