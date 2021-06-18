import numpy as np
import cv2

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import adamax_v2
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

emotion_model.compile(loss='categorical_crossentropy', optimizer=adamax_v2(lr=0.0001, decay=1e-6), metrics=['accuracy'])

emotion_detector = emotion_model.fit_generator(
                                                training_set,
                                                steps_per_epoch=28709,
                                                epochs=50,
                                                validation_data=test_set,
                                                validation_steps=7178
)


                                                