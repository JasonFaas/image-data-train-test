# Mulit-class logistic regression:
# Train classifier on each label separately AND
# AND use those to predict

# Going to use Stratified Shuffle Split to verify that all target categories are in train data
# Modified to handle multiple categories


import numpy as np
import cv2 as cv
import operator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from image_test_space import DisplayImage
from sklearn.linear_model import LogisticRegression

import matplotlib
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import tensorflow as tf

matplotlib.use("MacOSX")
from matplotlib import pyplot as plt
import glob

# Gather images to review
large_resources = '../../../../image-data-train-test-large-data/Coccidia/img/'
images_to_review = glob.glob(large_resources + "000*" + ".jpg")
screen_size = 96

# Get training data
img_mod = DisplayImage(screen_size=screen_size)
x_values, y_values = img_mod.get_training_values(images_to_review)

x_values = np.array(x_values)
x_train, x_test, y_train, y_test = train_test_split(x_values,
                                                    y_values,
                                                    test_size=0.2,
                                                    random_state=10)

# Cleanup data for CNN
x_train_v3 = x_train.astype(np.float32)
x_test_v3 = x_test.astype(np.float32)
x_train_v3 /= 255
x_test_v3 /= 255

input_shape = (screen_size, screen_size, 3)
review_failures = False

model = Sequential()
# TODO investigate kernel_size here
model.add(Conv2D(screen_size, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(2, activation=tf.nn.softmax))

# early_stopping_monitor = EarlyStopping(patience=10, monitor='acc')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping_monitor = EarlyStopping(patience=10, monitor='acc')
model.fit(np.array(x_train_v3),
          np.array(y_train),
          epochs=5,
          callbacks=[early_stopping_monitor],
          verbose=1)

predictions = model.predict(np.array(x_test_v3))
predictions = list(map(lambda v: np.argmax(np.array(v)), predictions))

# scoring info
if y_test is not None:
    total_test = len(y_test)
    predictions = np.array(predictions).astype(bool)

    nonzero = np.count_nonzero(np.array(predictions == y_test))
    # print("params:: layer_count: " + str(layer_count) + " " + "layer_size:" + str(layer_size))
    score = round(nonzero / total_test, 3)
    print("\tScore: " + str(score))
    # dataframe.at[layer_count, layer_size] = score

    if review_failures:
        guess_vs_actual = predictions == y_test
        for idx, good_guess in enumerate(guess_vs_actual):
            if not good_guess:
                print("Guess " + str(predictions[idx]) + " \tActual " + str(y_test[idx]))
                image = x_test[idx]
                print(image[screen_size * screen_size:])
                image = np.reshape(image[0:screen_size * screen_size], (-1, screen_size, 1))
                image.astype(np.uint8)
                cv.imshow("failure", image)
                if cv.waitKey(0) & 0xFF == ord('q'):
                    break
