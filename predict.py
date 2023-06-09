# to predict from images put images into input folder and we get output  saved in input/Test folder



from keras.models import load_model
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil as sh
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

print("[INFO] loading model...")
model = load_model('output/fire_detection.h5')

count = 0
images = []
labels = []
i = 0
lis1 = os.listdir('input/Test')
if len(lis1) > 0:
    sh.rmtree('input/Test')
    os.mkdir('input/Test')


def predictor(img1, tmodel):
    classes = ['Fire', 'Smoke', 'Non_Fire']
    imgp = cv2.resize(img1, (128, 128))
    imgp = imgp.astype('float32') / 255.
    predic = tmodel.predict(np.expand_dims(imgp, axis=0))[0]
    res = classes[np.argmax(predic)]
    hv = np.argmax(predic)
    return res

print("for single image prediction press 1 \n for batch prediction press 2")
inp = int(input())
if inp == 1:
    img = os.listdir('input/')[0]
    img = cv2.imread('input/'+ img)

    res = predictor(img,model)
    img = cv2.resize(img, (500, 400))
    cv2.putText(img, res, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 3)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit(0)



for img in os.listdir('input/'):
    print(f"File accessed {i}", img.title())
    title = img.title()
    if img.title() == 'Thumbs.db' or img.title() == None:  # added this line because prgrm keep crashing while accessing last images in non fire folder
        break
    if img.title() == 'Test':#Test is a folder
        continue
    img = cv2.imread('input/' + img)
    img_cp = img
    result = predictor(img,model)
    cv2.putText(img_cp, result, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 3)
    cv2.imwrite(f'input/Test/{title}.png', img_cp)
    i += 1
