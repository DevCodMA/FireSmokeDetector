import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
#from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization, Dense, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, \
    LeakyReLU
from keras.callbacks import ModelCheckpoint,EarlyStopping,CSVLogger,ReduceLROnPlateau


print("starting")
INIT_LR = 0.1
BATCH_SIZE = 64
NUM_EPOCHS = 50
lr_find = True
classes = ['Fire', 'Smoke', 'Non_Fire']

images = []
labels = []
count =0
for c in classes:
    #try:
        for img in os.listdir('Image Dataset/' + c):
            if img == 'Thumbs.db':# added this line because prgrm accessing windows tumbnail database and crashing
                continue
            img = cv2.imread('Image Dataset/' + c + '/' + img)


            print("directory accessed",count,c)
            img = cv2.resize(img, (128, 128))
            images.append(img)
            if c=='Fire':
                labels.append(0)
            elif c=='Smoke':
                labels.append(1)
            else:
                labels.append(2)

            count+=1
            #labels.append([0, 1][c == 'Fire'])
    #except:
        #print("failed")
        #pass

images = np.array(images, dtype='float32') / 255.

ind = np.random.randint(0,len(images))
cv2.imshow(str(labels[ind]),images[ind])
cv2.waitKey(0)
cv2.destroyAllWindows()

labels = np.array(labels)
labels = np_utils.to_categorical(labels,num_classes=3)
print("to category")
d = {}
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals
d[0] = classWeight[0]
d[1] = classWeight[1]
d[2] = classWeight[2]
print("test train split")
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.25, shuffle=True, random_state=42)
print("augmentation")
aug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

#model

model = Sequential()

# CONV => RELU => POOL
model.add(SeparableConv2D(32,(7,7),padding='same',input_shape=(128,128,3)))
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.05))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

# CONV => RELU => POOL
model.add(SeparableConv2D(32,(5,5),padding='same'))
model.add(LeakyReLU(alpha=0.05))
#model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

# CONV => RELU => CONV => RELU => POOL
model.add(SeparableConv2D(32,(3,3),padding='same'))
model.add(LeakyReLU(alpha=0.05))
#model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(SeparableConv2D(64,(3,3),padding='same'))
model.add(LeakyReLU(alpha=0.05))
#model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))



# first set of FC => RELU layers
model.add(Flatten())
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.05))
#model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


# second set of FC => RELU layers
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.05))
#model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
print("output node")
# softmax classifier
print(f"&&& {len(classes)}&&&&&&")
model.add(Dense(len(classes)))
model.add(Activation("softmax"))
print("optimization")
#opt = SGD(learning_rate=INIT_LR, momentum=0.9,decay=INIT_LR / NUM_EPOCHS)
#changed from SGD to Adam
model.compile(loss='categorical_crossentropy',
              optimizer="Adam",
              metrics=['accuracy'])

print(model.summary())

#implemented early stopping to avoid overfitting (even without it model failed to predict trianed image)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.2, patience=5,min_lr=0.01)
eStop = EarlyStopping(monitor='val_loss',patience=3,mode='min',verbose=1)
log_csv = CSVLogger('fireLogs.csv',separator=',',append=False)
checkpoint = ModelCheckpoint(filepath='best_model.h5',monitor='val_acc' ,save_best_only=True ,verbose=1,mode='max')
callbackList =[eStop, checkpoint, log_csv, reduce_lr]


#training
print("trining starting")
print("[INFO] training network...")
H = model.fit(
    aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
    epochs=NUM_EPOCHS,
    class_weight=d,
    callbacks=callbackList,
    verbose=1)
#graph
N = np.array(H.epoch)
plt.figure(figsize=(12,8))
plt.subplot(121)
plt.title("Losses")
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.subplot(122)
plt.title("Accuracies")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.legend()
plt.savefig("output/training_plot.png")

print("[INFO] serializing network to '{}'...".format('output/model'))

#saving
print("model going to  be saved")
model.save('output/fire_detection.h5')


