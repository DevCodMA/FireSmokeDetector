from flask import Flask, request, render_template, jsonify, Response, send_file
import numpy as np
from pickle import load, dump
from ultralytics import YOLO
import cv2
import numpy as np
from twilio.rest import Client
from time import time, sleep
import pygame
import os
import base64
import shutil
import tqdm
import threading

app = Flask(__name__)

start  = time()
cnt = True
def generate(phone):
    global start
    global cnt
    # set Twilio account SID, auth token, and phone number
    account_sid = 'AC70d82eba573132115cf2c557ac85bdb1'
    auth_token = 'd028c51cbc25a7b098cf788cd5dd4177'
    twilio_number = '+19893732989'

    # create Twilio client
    client = Client(account_sid, auth_token)

    # send SMS message containing OTP
    end = time()
    print(np.round(end-start, 0))
    if (np.round(end-start, 0) > 300) or cnt:
        message = client.messages.create(
            body=f'Fire detected!!!',
            from_=twilio_number,
            to="+91"+phone
        )
        start = time()
        cnt = False

model = YOLO(r"yolo_custom_w\combined yolo\model.pt")
cam_stat = False

def play_sound():
    pygame.init()
    sound = pygame.mixer.Sound('beep.wav')
    sound.play()
    pygame.time.wait(int(sound.get_length() * 750))
    pygame.quit()

def gen_frames(cap, conf, phoneno, alarm, sms):  
    while cam_stat:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model.predict(source=frame, show=False, task='detect', imgsz=640, conf=conf, save=False)
            col = (255, 255, 255)
            for result in results[0].boxes:
                confidence = result.conf.item()
                if result.cls == 0:
                    label = "Fire"
                    col = (0, 0, 255)
                    if confidence > conf+0.1:
                        if alarm == 'true':
                            sound_thread = threading.Thread(target=play_sound)
                            sound_thread.start()
                        if sms == 'true':
                            generate(phoneno)
                elif result.cls == 1:
                    label = "Smoke"
                    col = (255, 0, 0)
                else:
                    label = "Unknown"
                
                for box in results[0].boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        
@app.route("/", methods=["GET", "POST"])
def index():
    try:
        with open("settings.pkl", "rb") as file:
            data = load(file)
        if len(data) != 5:
            return render_template("index.html", error=True)
        else:
            return render_template("index.html")
    except:
        return render_template("index.html", error=True)

@app.route("/settings", methods=["GET", "POST"])
def settings():
    conf = float(request.form['confidence'])/100
    phone = request.form['phoneno']
    src = request.form['camera']
    alarm = request.form['alarm']
    sms = request.form['sms']
    ls = [conf, phone, src, alarm, sms]
    with open("settings.pkl", "wb") as file:
        dump(ls, file)
    
    return "Settings Saved!"

@app.route("/video_feed")
def video_feed():
        with open("settings.pkl", "rb") as file:
            data = load(file)
        conf, phoneno, cam, alarm, sms = data
        vidcap = cv2.VideoCapture(int(cam))
        return Response(gen_frames(vidcap, conf, phoneno, alarm, sms), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/control_cam", methods=["GET", "POST"])
def control():
    global cam_stat
    print(type(request.form['status']))
    if request.form['status'] == 'true':
        cam_stat = True
    else:
        cam_stat = False
    video_feed()
    return "status_good"

@app.route("/analyze_data", methods=["GET", "POST"])
def process_file():
    img = ""
    with open("settings.pkl", "rb") as file:
        data = load(file)
    conf, phoneno, cam, alarm, sms = data
    file = request.files['uploadfiledata']
    ftype = request.form['filetype']
    fname = f'Uploads/{file.filename}'
    file.save(fname)
    if ftype == 'video':
        cap = cv2.VideoCapture(r"".join(fname))
        fourcc = cv2.VideoWriter_fourcc(*'avci')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(r"Processed/output.mp4", fourcc, 30.0, (width, height), True)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                results = model.predict(source=frame, show=False, task='detect', imgsz=640, conf=conf, save=False)
                col = (255, 255, 255)
                for result in results[0].boxes:
                    confidence = result.conf.item()
                    if result.cls == 0:
                        label = "Fire"
                        col = (0, 0, 255)
                    elif result.cls == 1:
                        label = "Smoke"
                        col = (255, 0, 0)
                    else:
                        label = "Unknown"
                                
                    for box in results[0].boxes.xyxy:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3)
                        cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
                out.write(frame)
            else:
                break
        cap.release()
        out.release()
        os.remove(r"".join(fname))
        if os.path.exists(r"Processed\output.mp4"):
            return "true"
        else:
            return "false"

    elif ftype == 'image':
        img = cv2.imread(r"".join(fname))
        results = model.predict(source=img, show=False, task='detect', imgsz=640, conf=conf, save=True)       
        if os.path.exists(r"".join(fname)):
            os.remove(r"".join(fname))
        if os.path.exists(r"runs\detect"):
            img = cv2.imread(r'runs\detect\predict\image0.jpg')
            ret, buf = cv2.imencode('.jpg', img)
            img = buf.tobytes()
            img = f"data:image/jpeg;base64,{base64.b64encode(img).decode('utf-8')}"
            shutil.rmtree('runs')
        return img
    
@app.route("/video_data")
def send_vdata():
    return send_file(r"Processed\output.mp4", mimetype='video/mp4', as_attachment=False)

if __name__ == "__main__":
    app.run(debug=True)