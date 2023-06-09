from ultralytics import YOLO
import cv2
import numpy as np
from twilio.rest import Client
from time import time, sleep
import pygame

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

#model = YOLO('C:\\Users\sudo\PycharmProjects\pythonProject\\runs\detect\\train3\weights\\last.pt')
# model = YOLO("C:\\Users\sudo\PycharmProjects\pythonProject\yolo_custom_w\\best.pt")
model = YOLO(r"C:\Users\majma\Downloads\Fire and Smoke Detetcion\yolo_custom_w\combined yolo\best.pt")
source =r"output.jpg"
# source = cv2.resize(cv2.imread(source), (640, 640))
# cv2.imwrite("output.jpg", np.array(source))
# img = BytesIO()

# with open(img, 'wb') as file:
#     file.write(source)



# results = model.predict(source=source, show=True)


# x,y,h = res[0].boxes.boxes

# img = cv2.imread(res[0].orig_img)
# cv2.rectangle(img, x, y)
# cv2.imshow("Frame",img)
# cv2.waitKey(0)


vidcap = cv2.VideoCapture(0)
vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

if vidcap.isOpened():
    ret, frame = vidcap.read()
    if ret:
        while (True):
            ret, img = vidcap.read()
            res = model.predict(source=img, show=True, task='detect', imgsz=640, conf=0.5, save=False,verbose = True)
            # cnf = np.array(res[0].boxes.conf)
            # if cnf.size != 0:
            #     if np.round(np.max(cnf), 2) > 0.6:
            #         pygame.init()

            #         # Load the sound file
            #         sound = pygame.mixer.Sound('beep.wav')

            #         # Play the sound
            #         sound.play()

            #         # Wait for the sound to finish playing
            #         pygame.time.wait(int(sound.get_length() * 1000))

            #         # Quit Pygame
            #         pygame.quit()
            #         generate('7034834276')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        print("Error : Failed to capture frame")
else:
    print("Cannot open camera")



# res = model.predict(source=r"C:\Users\majma\Downloads\Fire and Smoke Detetcion\output.jpg", show=True, task='detect', imgsz=640, conf=0.20, save=False,verbose = True)
# print(res[0].boxes)
# # clist = res[0].boxes.cls
# # cls = set()
# # for cno in clist:
# #     cls.add(model.names[int(cno)])


# # print(list(cls))



