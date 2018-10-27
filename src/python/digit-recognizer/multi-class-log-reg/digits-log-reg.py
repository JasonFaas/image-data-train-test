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


resources = '../../../resources/digit-recognizer'
train_csv = 'train.csv'
# train_csv = 'jason_train_10000.csv'
# train_csv = 'jason_train_5000.csv'
# train_csv = 'jason_train_4000.csv'
# train_csv = 'jason_train_2000.csv'
# train_csv = 'jason_train_1000.csv'
csv_filename = '%s/%s' % (resources, train_csv)

# read training info
digit_train_set = pd.read_csv(csv_filename)
image_info = DisplayImage(csv_filename)
digit_train_set = image_info.get_all_info()

# separate training info into samples and target
samples_v1 = digit_train_set[:, 1]
target = digit_train_set[:, 0]
target = target.astype(int)

# print(type(target[0]))
# print(target)
# exit(0)

samples_v2 = list(map(lambda v: np.reshape(v, (-1)), samples_v1))
samples_v3 = []


def circle_count(img):
    ff_mean = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, -2)
    cv.floodFill(ff_mean, np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8), (0, 0), 255)

    ff_mean_inv = cv.bitwise_not(ff_mean)

    cnts = cv.findContours(ff_mean_inv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]

    result = np.zeros((28, 28, 1), np.uint8)

    # for c in cnts:
    #     M = cv.moments(c)
    #     if M["m00"] > .1:
    #         cX = int(M["m10"] / M["m00"])
    #         cY = int(M["m01"] / M["m00"])
    #     else:
    #         halfway_c = int(c.shape[0] - 1 / 2)
    #         cX = c[halfway_c, 0, 0]
    #         cY = c[halfway_c, 0, 1]
    #
    #     result[cY, cX] = 255

    return len(cnts)


for idx, sample in enumerate(samples_v2):
    new_sample = np.zeros((sample.shape[0] + 1), np.uint8)
    new_sample[0:sample.shape[0]] = sample[:]

    # include circle count
    new_sample[-1] = circle_count(samples_v1[idx])
    samples_v3.append(new_sample)




x_train, x_test, y_train, y_test = train_test_split(samples_v3,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=10)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# TODO try out other c_param after setting up points of circles
for c_param in [0.1]:
    for penalty in ['l2']:
        clf = LogisticRegression(penalty=penalty, C=c_param)
        clf.fit(x_train, y_train)

        print("\nC " + str(c_param))
        print("P " + str(penalty))
        print(round(clf.score(x_test, y_test), 3))

