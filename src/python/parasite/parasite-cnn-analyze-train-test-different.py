# Mulit-class logistic regression:
# Train classifier on each label separately AND
# AND use those to predict

# Going to use Stratified Shuffle Split to verify that all target categories are in train data
# Modified to handle multiple categories


import numpy as np
import cv2 as cv
import pandas as pd
from sklearn.model_selection import train_test_split
from image_test_space import DisplayImage

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
images_to_review = glob.glob(large_resources + "03*" + ".jpg")

train_images, test_images, _, _ = train_test_split(images_to_review, range(0, len(images_to_review)), test_size=0.2, random_state=91)

screen_size = 96

# Get training data
img_size = 512
img_mod = DisplayImage(img_size=img_size, screen_size=screen_size)
x_values, y_values = img_mod.get_training_values(train_images)

x_train = np.array(x_values)
y_train = y_values

# Cleanup data for CNN
x_train_v3 = x_train.astype(np.float32)
x_train_v3 /= 255

input_shape = (screen_size, screen_size, 3)
review_failures = True

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


x_all_test_values = []
y_all_test_values = []
for img_filename in test_images:
    img = cv.imread(img_filename)
    positive_mask = img_mod.get_positive_mask(img_filename)

    x_img_test_values = []
    y_img_test_values = []

    for xmin in range(0, img_size, 64):
        for ymin in range(0, img_size, 64):
            xmax = xmin + screen_size
            ymax = ymin + screen_size
            if xmax > img_size:
                xmax = img_size
                xmin = xmax - screen_size
            if ymax > img_size:
                ymax = img_size
                ymin = ymax - screen_size

            img_roi = img[ymin:ymax, xmin:xmax]
            positive_mask_nonzero = np.count_nonzero(positive_mask[ymin:ymax, xmin:xmax])
            y_all_test_values.append(positive_mask_nonzero > 300)
            x_all_test_values.append(img_roi)
            y_img_test_values.append(positive_mask_nonzero > 300)
            x_img_test_values.append(img_roi)


    if review_failures:
        x_test = np.array(x_img_test_values)
        x_test_v2 = x_test.astype(np.float32)
        x_test_v2 /= 255

        predictions = model.predict(np.array(x_test_v2))
        predictions = list(map(lambda v: np.argmax(np.array(v)), predictions))
        predictions = np.array(predictions).astype(bool)

        idx = -1
        img_display = img.copy()
        for xmin in range(0, img_size, 64):
            for ymin in range(0, img_size, 64):
                idx += 1
                xmax = xmin + screen_size
                ymax = ymin + screen_size
                if xmax > img_size:
                    xmax = img_size
                    xmin = xmax - screen_size
                if ymax > img_size:
                    ymax = img_size
                    ymin = ymax - screen_size
                if predictions[idx] and y_img_test_values[idx]:
                    cv.rectangle(img_display, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)
                if predictions[idx] and not y_img_test_values[idx]:
                    cv.rectangle(img_display, (xmin, ymin), (xmax, ymax), (0, 255, 255), thickness=2)
                if not predictions[idx] and y_img_test_values[idx]:
                    cv.rectangle(img_display, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)

        cv.imshow("img_display", img_display)
        if cv.waitKey(0) & 0xFF == ord('q'):
            break

x_test = np.array(x_all_test_values)
x_test_v2 = x_test.astype(np.float32)
x_test_v2 /= 255

predictions = model.predict(np.array(x_test_v2))
predictions = list(map(lambda v: np.argmax(np.array(v)), predictions))
predictions = np.array(predictions).astype(bool)

y_test = y_all_test_values

# scoring info
if y_test is not None:
    total_test = len(y_test)

    nonzero = np.count_nonzero(np.array(predictions == y_test))
    # print("params:: layer_count: " + str(layer_count) + " " + "layer_size:" + str(layer_size))
    score = round(nonzero / total_test, 3)
    print("\tScore: " + str(score))
    # dataframe.at[layer_count, layer_size] = score
