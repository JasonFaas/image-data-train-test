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



small_resources = '../../resources/parasite/label/'
large_resources = '../../../../image-data-train-test-large-data/Coccidia/img/'


images_to_review = glob.glob(large_resources + "000*" + ".jpg")

x_values = []
y_values = []


img_mod = DisplayImage()
        
screen_size = 96
min_gaus_nonzeros = screen_size ** 2

for idx, img_filename in enumerate(images_to_review):
    base_name = img_filename[-8:-4]
    # print(base_name)
    xml_filename = small_resources + base_name + ".xml"

    img = cv.imread(img_filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


    screen_pad = int(screen_size / 10)
    block = 251
    C = 45

    gaus = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, block, C)
    positive_mask = np.zeros(gray.shape)

    # Log True values and build positive mask
    for rect_idx in range(img_mod.get_rectangle_count(xml_filename)):
        xmin_org, ymin_org, xmax_org, ymax_org = img_mod.get_data_from_xml(xml_filename,
                                                                           rect_idx)

        cv.rectangle(positive_mask, (xmin_org, ymin_org), (xmax_org, ymax_org), (255), -1)

        for corner in ["tl", "tr", "bl", "br"]:
            xmin, xmax, ymin, ymax = img_mod.get_roi(xmin_org, xmax_org, ymin_org, ymax_org, screen_pad, screen_size, corner)
            if xmin < 0 or xmax >= 512 or ymin < 0 or ymax >= 512:
                xmin, xmax, ymin, ymax = img_mod.get_roi(xmin_org, xmax_org, ymin_org, ymax_org, 0, screen_size, corner)

            if 0 <= xmin < xmax < 512 and 0 <= ymin < ymax < 512:
                x_values.append(img[ymin:ymax, xmin:xmax])
                y_values.append(True)
                nonzero = np.count_nonzero(gaus[ymin:ymax, xmin:xmax])
                if nonzero < min_gaus_nonzeros:
                    min_gaus_nonzeros = nonzero

    # cv.imshow("mask", positive_mask)
    # cv.waitKey(0)

    # Log False values based on gaus and positive_mask
    for xmin in range(0, 512, 64):
        for ymin in range(0, 512, 64):
            xmax = xmin + screen_size
            ymax = ymin + screen_size
            if xmax > 512:
                xmax = 512
                xmin = xmax - screen_size
            if ymax > 512:
                ymax = 512
                ymin = ymax - screen_size
            gaus_nonzero = np.count_nonzero(gaus[ymin:ymax, xmin:xmax])
            positive_mask_nonzero = np.count_nonzero(positive_mask[ymin:ymax, xmin:xmax])

            if gaus_nonzero > 100 and positive_mask_nonzero == 0:
                x_values.append(img[ymin:ymax, xmin:xmax])
                y_values.append(False)


x_values = np.array(x_values)
x_train, x_test, y_train, y_test = train_test_split(x_values,
                                                    y_values,
                                                    test_size=0.2,
                                                    random_state=10)
# x_train_v2 = np.array(x_train).reshape(len(x_train), screen_size, screen_size, 1)
# x_test_v2 = np.array(x_test).reshape(len(x_test), screen_size, screen_size, 1)

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
