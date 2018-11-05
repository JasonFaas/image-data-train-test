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
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import tensorflow as tf

from keras.models import load_model

matplotlib.use("MacOSX")
from matplotlib import pyplot as plt
import glob

print("Is there a GPU available: "),
print(tf.test.is_gpu_available())

review_failures = True
new_model = False

# Gather images to review
large_resources = '../../../../image-data-train-test-large-data/Coccidia/img/'
base_train = "0"
images_to_review = glob.glob(large_resources + base_train + "*" + ".jpg")

train_images, test_images, _, _ = train_test_split(images_to_review,
                                                   range(0, len(images_to_review)),
                                                   test_size=0.2,
                                                   random_state=10)

screen_size = 32
img_size = 512
zeros_mask = np.zeros((img_size, img_size, 1), dtype=np.uint8)
img_mod = DisplayImage(img_size=img_size, screen_size=screen_size)

model_save_name = "model_save_v2_%sx.h5" % base_train
if new_model:
    # Get training data
    x_values, y_values = img_mod.get_training_values(train_images)

    x_train = np.array(x_values)
    y_train = y_values

    # Cleanup data for CNN
    x_train_v3 = x_train.astype(np.float32)
    x_train_v3 /= 255

    y_train_array = np.array(y_train)
    print("Total samples:\t\t" + str(y_train_array.shape[0]))
    print("Parasite samples:\t" + str(np.count_nonzero(y_train_array)))

    input_shape = (screen_size, screen_size, 3)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8, kernel_size=(3,3), input_shape=input_shape),

        tf.keras.layers.Conv2D(8, kernel_size=(3,3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        tf.keras.layers.Conv2D(16, kernel_size=(3,3), activation=tf.nn.relu),
        tf.keras.layers.Conv2D(16, kernel_size=(3,3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        #
        # tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation=tf.nn.relu),
        # tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation=tf.nn.relu),
        # tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        # tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation=tf.nn.relu),
        # tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation=tf.nn.relu),
        # tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    early_stopping_monitor = EarlyStopping(patience=10, monitor='acc')
    model.fit(np.array(x_train_v3),
              y_train_array,
              epochs=20,
              callbacks=[early_stopping_monitor],
              verbose=1)

    model.save(model_save_name)
else:
    model = load_model(model_save_name)



x_all_test_values = []
y_all_test_values = []
for img_filename in test_images:
    img = cv.imread(img_filename)
    positive_mask = img_mod.get_positive_mask(img_filename)
    good_result = np.zeros(positive_mask.shape, dtype=positive_mask.dtype)
    bad_result = np.zeros(positive_mask.shape, dtype=positive_mask.dtype)

    y_img_nuance = []
    x_img_test_values = []
    y_img_test_values = []

    test_step_size = int(screen_size / 2)
    for xmin in range(0, img_size, test_step_size):
        for ymin in range(0, img_size, test_step_size):
            xmax = xmin + screen_size
            ymax = ymin + screen_size
            if xmax > img_size or ymax > img_size:
                print("Out of bounds (expected)")
                continue

            img_roi = img[ymin:ymax, xmin:xmax]
            positive_mask_nonzero = np.count_nonzero(positive_mask[ymin:ymax, xmin:xmax])
            y_value = positive_mask_nonzero > screen_size ** 2 / 3
            y_all_test_values.append(y_value)
            x_all_test_values.append(img_roi)
            y_img_nuance.append(positive_mask_nonzero > screen_size ** 2 / 10)
            y_img_test_values.append(y_value)
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
        img_display_2 = img.copy()
        for xmin in range(0, img_size, test_step_size):
            for ymin in range(0, img_size, test_step_size):
                idx += 1
                xmax = xmin + screen_size
                ymax = ymin + screen_size
                if xmax > img_size or ymax > img_size:
                    print("Out of bounds (expected review)")
                    idx -= 1
                    continue
                if predictions[idx] and y_img_test_values[idx]:
                    cv.rectangle(img_display, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)
                    cv.rectangle(good_result, (xmin, ymin), (xmax, ymax), (255), thickness=-1)
                if predictions[idx] and not (y_img_test_values[idx] or y_img_nuance[idx]):
                    cv.rectangle(img_display, (xmin, ymin), (xmax, ymax), (0, 255, 255), thickness=2)
                    cv.rectangle(bad_result, (xmin, ymin), (xmax, ymax), (255), thickness=-1)
                if not predictions[idx] and y_img_test_values[idx]:
                    cv.rectangle(img_display, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)


        cv.imshow("img_display", img_display)
        cv.imshow("positive_mask", positive_mask)
        cv.imshow("result_zeros", good_result)
        good_ex = cv.morphologyEx(good_result, cv.MORPH_GRADIENT, np.ones((5, 5)))
        bad_ex = cv.morphologyEx(bad_result, cv.MORPH_GRADIENT, np.ones((5, 5)))
        cv.imshow("5_final", cv.add(img_display_2, cv.merge((zeros_mask, good_ex, bad_ex))))
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
