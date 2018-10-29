# Mulit-class logistic regression:
# Train classifier on each label separately AND
# AND use those to predict

# Going to use Stratified Shuffle Split to verify that all target categories are in train data
# Modified to handle multiple categories

import sys
sys.path.insert(0, '/Users/jasonfaas/Code/image-data-train-test/src/python/digit-recognizer/multi-class-log-reg')

import numpy as np
import cv2 as cv
import operator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from image_test_space import DisplayImage
from sklearn.linear_model import LogisticRegression


resources = '../../../../resources/digit-recognizer'
# train_csv = 'train.csv'
# train_csv = 'jason_train_10000.csv'
train_csv = 'jason_train_5000.csv'
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

samples_v2 = np.array(list(map(lambda v: np.reshape(v, (28, 28, 1)), samples_v1)))
samples_v2 = samples_v2.astype(np.uint8)

target_value = 8
samples_v2 = samples_v2[target == target_value]
target = target[target == target_value]

for img in samples_v2:
    cv.imshow("what", img)
    if cv.waitKey(0) & 0xFF == ord('q'):
        break

print(samples_v2.shape[0])

for min_dist in range(8, 9, 1):
        for param_2 in range(8, 9, 1):
            two_circles = 0
            for img in samples_v2:
                circles = cv.HoughCircles(img,
                                          cv.HOUGH_GRADIENT,
                                          1,
                                          minDist=min_dist,
                                          param1=30,
                                          param2=param_2,
                                          minRadius=1,
                                          maxRadius=10)
                if not (circles is None or len(circles[0, :]) != 2):
                    two_circles += 1
                else:
                    im2, contours, hierarchy = cv.findContours(img, 1, 2)
                    cnt = contours[0]
                    epsilon = 0.01 * cv.arcLength(cnt, True)
                    approx = cv.approxPolyDP(cnt, epsilon, True)
                    print("\n")
                    print(epsilon)
                    print(approx)

                    cv.imshow("fail", img)
                    cv.waitKey(0)

            if two_circles > 350:
                print("\nparam_2: " + str(param_2) + " \tmin_dist: " + str(min_dist))
                print(two_circles)
