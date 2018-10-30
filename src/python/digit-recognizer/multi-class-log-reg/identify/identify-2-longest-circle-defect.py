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
# train_csv = 'jason_train_5000.csv'
# train_csv = 'jason_train_4000.csv'
# train_csv = 'jason_train_2000.csv'
train_csv = 'jason_train_1000.csv'
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

target_value = 6
samples_v2 = samples_v2[target == target_value]
target = target[target == target_value]


def two_points_distance(pt_a, pt_b):
    return ((pt_a[0] - pt_b[0]) ** 2 + (pt_a[1] - pt_b[1]) ** 2) ** 0.5


for img in samples_v2:

    _, thresh = cv.threshold(img, 10, 255, cv.THRESH_BINARY)

    zeros_start = np.zeros(img.shape, np.uint8)
    zeros_end = np.zeros(img.shape, np.uint8)

    im2, contours, hierarchy = cv.findContours(img, 2, 1)
    cnt = contours[0]

    hull = cv.convexHull(cnt, returnPoints=False)
    defects = cv.convexityDefects(cnt, hull)



    if not(defects is None):
        furthest_idx = 0
        furthest_dist = 0
        defect_count = defects.shape[0]
        for idx in range(defect_count):
        # for idx in range(1):
            s, e, f, d = defects[idx, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            print(cnt[s][0])
            print(cnt[e][0])
            print(cnt[f][0])
            print(d)

            a = two_points_distance(start, far)
            b = two_points_distance(end, far)
            c = two_points_distance(start, end)
            distance = a * b / c
            print(distance)
            print("")

            if distance > furthest_dist:
                furthest_dist = distance
                furthest_idx = idx


        s, e, f, d = defects[furthest_idx, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        cv.line(zeros_start, start, end, (int(180 * idx / defect_count) + 70), 1)
        cv.circle(zeros_start, far, 1, [int(180 * idx / defect_count) + 70], -1)

    cv.imshow("what", img)
    cv.imshow("zeros_start", zeros_start)
    cv.imshow("zeros_end", zeros_end)
    if cv.waitKey(0) & 0xFF == ord('q'):
        break
#
# for min_dist in range(12, 50, 2):
#     for param_1 in range(10, 100, 20):
#         circle_counts = []
#         circle_x_s = []
#         circle_y_s = []
#         circle_r_s = []
#         for img in samples_v2:
#             circles = cv.HoughCircles(img,
#                                       cv.HOUGH_GRADIENT,
#                                       1,
#                                       minDist=min_dist,
#                                       param1=param_1,
#                                       param2=10,
#                                       minRadius=1,
#                                       maxRadius=10)
#             if circles is None:
#                 circle_counts.append(0)
#             else:
#                 circle_count = len(circles[0, :])
#                 circle_counts.append(circle_count)
#                 first_circle = circles[0, 0]
#                 circle_x_s.append(first_circle[0])
#                 circle_y_s.append(first_circle[1])
#                 circle_r_s.append(first_circle[2])
#
#         average_circles = np.average(circle_counts)
#         average_x = round(np.average(circle_x_s), 2)
#         average_y = round(np.average(circle_y_s), 2)
#         average_r = round(np.average(circle_r_s), 2)
#
#         if 1.01 > average_circles > .90:
#             print("\nGood")
#             print(str(average_x) + " " + str(average_y) + " " + str(average_r))
#         else:
#             print("\nBad")
#             print("min_dist: " + str(min_dist) + " \tparam_1: " + str(param_1))
#             print(np.round(average_circles, 2))
#
