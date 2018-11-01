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
matplotlib.use("MacOSX")
from matplotlib import pyplot as plt
import glob


def get_single_data_from_xml(tag, idx):
    return int(mydoc.getElementsByTagName(tag)[idx].firstChild.data)

small_resources = '../../resources/parasite/label/'
large_resources = '../../../../image-data-train-test-large-data/Coccidia/img/'


images_to_review = glob.glob(large_resources + "00*" + ".jpg")

x_train = []
y_train = []


def get_roi(xmin_org, xmax_org, ymin_org, ymax_org, pad, size, param):
    if param[0] == 't':
        xmin = xmin_org - pad
        xmax = xmin + size
    else:
        xmax = xmax_org + pad
        xmin = xmax - size
        
    if param[1] == 'l':
        ymin = ymin_org - pad
        ymax = ymin + size
    else:
        ymax = ymax_org + pad
        ymin = ymax - size
    return xmin, xmax, ymin, ymax
        
        
min_gaus_nonzeros = 96 * 96

for idx, img_filename in enumerate(images_to_review):
    base_name = img_filename[-8:-4]
    # print(base_name)
    xml_filename = small_resources + base_name + ".xml"

    img = cv.imread(img_filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    from xml.dom import minidom
    mydoc = minidom.parse(xml_filename)


    screen_size = 96
    screen_pad = int(screen_size / 10)
    block = 251
    C = 45

    gaus = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, block, C)
    positive_mask = np.zeros(gray.shape)

    # Log True values and build positive mask
    for rect_idx in range(len(mydoc.getElementsByTagName('xmin'))):
        xmin_org = get_single_data_from_xml('xmin', rect_idx)
        ymin_org = get_single_data_from_xml('ymin', rect_idx)
        xmax_org = get_single_data_from_xml('xmax', rect_idx)
        ymax_org = get_single_data_from_xml('ymax', rect_idx)

        cv.rectangle(positive_mask, (xmin_org, ymin_org), (xmax_org, ymax_org), (255), -1)

        for corner in ["tl", "tr", "bl", "br"]:
            xmin, xmax, ymin, ymax = get_roi(xmin_org, xmax_org, ymin_org, ymax_org, screen_pad, screen_size, corner)
            if xmin < 0 or xmax >= 512 or ymin < 0 or ymax >= 512:
                xmin, xmax, ymin, ymax = get_roi(xmin_org, xmax_org, ymin_org, ymax_org, 0, screen_size, corner)

            if 0 <= xmin < xmax < 512 and 0 <= ymin < ymax < 512:
                x_train.append(img[ymin:ymax, xmin:xmax])
                y_train.append(True)
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

            if gaus_nonzero > 50 and positive_mask_nonzero == 0:
                x_train.append(img[ymin:ymax, xmin:xmax])
                y_train.append(False)


print(min_gaus_nonzeros)
array = np.array(x_train)
print(array.shape)




exit(0)


# separate training info into samples and target
# samples_v1 = digit_train_set[:, 1]
# target = digit_train_set[:, 0]
# target = target.astype(int)
#
#
# samples_v2 = list(map(lambda v: np.reshape(v, (-1)), samples_v1))
# samples_v3 = image_info.circle_info_arr(samples_v2, samples_v1)




x_train, x_test_before, y_train, y_test = train_test_split(samples_v3,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=10)
