import numpy as np
import cv2

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
# from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

train_image_dir = "faces/train"
test_image_dir = "faces/test"

################################ preprocessing Data ######################################

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(
                                                train_image_dir,
                                                target_size = (48, 48),
                                                batch_size = 64,
                                                color_mode = "grayscale",
                                                class_mode = "categorical"
                                                )

test_set = test_datagen.flow_from_directory(
                                            test_image_dir,
                                            target_size = (48, 48),
                                            batch_size = 64,
                                            color_mode = "grayscale",
                                            class_mode = "categorical"
                                            )

##########################################################################################

################################### Building Convolutional Neural Network ####################

emotion_model = Sequential()
emotion_model.add(Conv2D(32, (3, 3), activation='relu', input_shape = [48, 48, 1]))
emotion_model.add(Conv2D(32, (3, 3), activation='relu', input_shape = [48, 48, 1]))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation="relu"))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation="softmax"))

###############################################################################################

#################################### Compiling CNN ############################################

emotion_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

# emotion_detector = emotion_model.fit_generator(
#                                                 training_set,
#                                                 steps_per_epoch=28709,
#                                                 epochs=50,
#                                                 validation_data=test_set,
#                                                 validation_steps=7178
# )

emotion_detector = emotion_model.fit_generator(
                                                training_set,
                                                steps_per_epoch=100,
                                                epochs=50,
                                                validation_data=test_set,
                                                validation_steps=7178
)
emotion_model.save_weights('model.h5')

############################### Detecting Face From Webcam ############################

cv2.ocl_Device.setUseOpenCL(False)
emotion_dict = {
    0 : "Angry",
    1 : "Disgusted",
    2 : "Fearful",
    3 : "Happy",
    4 : "Neutral",
    5 : "Sad",
    6 : "Surprised"
}

cap_vid = cv2.VideoCapture(0)

while(True) :

    ret, frame = cap_vid.read()

    if (not ret) :
        break

    bounding_box = cv2.CascadeClassifier('faces/faces_deect/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultipleScale(gray_frame, scale_factor = 1.3, minNeighbors = 5)

    for (x, y, w, h) in num_faces :
        cv2.Rec




                                                