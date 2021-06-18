import numpy as np
import cv2

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
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
                                                color_mode = "gray_framescale",
                                                class_mode = "categorical"
                                                )

test_set = test_datagen.flow_from_directory(
                                            test_image_dir,
                                            target_size = (48, 48),
                                            batch_size = 64,
                                            color_mode = "gray_framescale",
                                            class_mode = "categorical"
                                            )

##########################################################################################

################################### Building Convolutional Neural Network ####################

emotion_model = Sequential()


