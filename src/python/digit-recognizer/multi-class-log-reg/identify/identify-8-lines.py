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
train_csv = 'jason_train_10000.csv'
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




samples_v2 = np.array(list(map(lambda v: np.reshape(v, (28, 28, 1)), samples_v1)))
samples_v2 = samples_v2.astype(np.uint8)

target_value = 8
samples_v2 = samples_v2[target == target_value]
target = target[target == target_value]


print(samples_v2.shape[0])

pad = 2
h_kernel = np.array([[1, 1], [-1, -1]], np.int32)
z_kernel = np.array([[1, -1], [1, -1]], np.int32)
hardcoded_kernel_size = 9
for blocksize in [9]:
    for C in [-2]:
        for img in samples_v2:
            # ff_copy = img.copy()
            # ff_mean = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blocksize, C)

            horizontal = np.zeros(img.shape, np.uint8)
            vertical = np.zeros(img.shape, np.uint8)
            for x in range(0, img.shape[0] - pad - 1):
                for y in range(0, img.shape[0] - pad - 1):
                    roi = img[y:y+pad, x:x+pad]
                    h = np.vdot(h_kernel, roi)
                    z = np.vdot(z_kernel, roi)
                    horizontal[y, x] = max(min(h, 255), 0)
                    vertical[y, x] = max(min(z, 255), 0)

            # horizontal = cv.subtract(horizontal, img)
            cv.imshow("img", img)
            cv.imshow("horizontal", horizontal)
            cv.imshow("vertical", vertical)
            cv.imshow("both", cv.bitwise_and(vertical, horizontal))
            if cv.waitKey(0) & 0xFF == ord('q'):
                exit(0)