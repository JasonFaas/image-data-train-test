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



# print(type(target[0]))
# print(target)
# exit(0)

samples_v2 = np.array(list(map(lambda v: np.reshape(v, (28, 28, 1)), samples_v1)))
samples_v2 = samples_v2.astype(np.uint8)

target_value = 8
samples_v2 = samples_v2[target == target_value]
target = target[target == target_value]


print(samples_v2.shape[0])


for blocksize in [7]:
    for C in [-3]:
        few_contours = 0
        dict_areas = {}
        dict_areas[0] = 0
        dict_areas[1] = 0
        dict_areas[2] = 0
        dict_areas[3] = 0
        dict_areas[4] = 0
        dict_areas[5] = 0
        for img in samples_v2:
            # ff_copy = img.copy()
            ff_mean = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blocksize, C)
            cv.floodFill(ff_mean, np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8), (0,0), 255)

            ff_mean_inv = cv.bitwise_not(ff_mean)

            cnts = cv.findContours(ff_mean_inv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # TODO this might be cnts[0]
            cnts = cnts[1]

            result = np.zeros((28, 28, 1), np.uint8)


            for c in cnts:
                M = cv.moments(c)
                if M["m00"] > .1:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    few_contours += 1
                    halfway_c = int(c.shape[0] - 1 / 2)
                    cX = c[halfway_c, 0, 0]
                    cY = c[halfway_c, 0, 1]

                result[cY, cX] = 255

            dict_areas[len(cnts)] += 1

            cv.imshow("ff_mean", ff_mean)
            cv.imshow("original_img", img)
            cv.imshow("result", result)
            if cv.waitKey(0) & 0xFF == ord('q'):
                break


        if dict_areas[2] > samples_v2.shape[0] / 2:
            print("\nC and blocksize:" + str(C) + " " + str(blocksize))
            print("Few Contours:" + str(few_contours))
            print(dict_areas)