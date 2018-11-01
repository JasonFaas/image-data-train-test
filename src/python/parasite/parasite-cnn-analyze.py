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

for idx, img_filename in enumerate(images_to_review):
    base_name = img_filename[-8:-4]
    print(base_name)
    xml_filename = small_resources + base_name + ".xml"

    img = cv.imread(img_filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    from xml.dom import minidom
    mydoc = minidom.parse(xml_filename)


    screen_size = 96
    screen_pad = int(screen_size / 10)
    block = 301
    C = 50
    print("\n")
    print(block)
    print(C)

    gaus = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, block, C)
    # for rect_idx in range(len(mydoc.getElementsByTagName('xmin'))):
    for rect_idx in range(1):
        xmin_org = get_single_data_from_xml('xmin', rect_idx)
        ymin_org = get_single_data_from_xml('ymin', rect_idx)
        xmax_org = get_single_data_from_xml('xmax', rect_idx)
        ymax_org = get_single_data_from_xml('ymax', rect_idx)
        # cv.rectangle(gaus, (xmin, ymin), (xmax, ymax), (255))

        # tl_roi
        xmin = xmin_org - screen_pad
        xmax = xmin + screen_size
        ymin = ymin_org - screen_pad
        ymax = ymin + screen_size
        if 0 <= xmin < xmax < 512 and 0 <= ymin < ymax < 512:
            cv.rectangle(gaus, (xmin, ymin), (xmax, ymax), (255))

        # bl_roi
        xmin = xmin_org - screen_pad
        xmax = xmin + screen_size
        ymax = ymax_org + screen_pad
        ymin = ymax - screen_size
        if 0 <= xmin < xmax < 512 and 0 <= ymin < ymax < 512:
            cv.rectangle(gaus, (xmin, ymin), (xmax, ymax), (255))

        # tr_roi
        xmax = xmax_org + screen_pad
        xmin = xmax - screen_size
        ymin = ymin_org - screen_pad
        ymax = ymin + screen_size
        if 0 <= xmin < xmax < 512 and 0 <= ymin < ymax < 512:
            cv.rectangle(gaus, (xmin, ymin), (xmax, ymax), (255))

        # br_roi
        xmax = xmax_org + screen_pad
        xmin = xmax - screen_size
        ymax = ymax_org + screen_pad
        ymin = ymax - screen_size
        if 0 <= xmin < xmax < 512 and 0 <= ymin < ymax < 512:
            cv.rectangle(gaus, (xmin, ymin), (xmax, ymax), (255))

        # c_roi
        
    print(img.shape)



    cv.imshow("zeros", img)
    cv.imshow("gray", gray)
    cv.imshow("gaus", gaus)
    key = cv.waitKey(0) & 0xFF
    if key == ord('q'):
        break


exit(0)

# read training info
digit_train_set = pd.read_csv(csv_filename)
image_info = DisplayImage(csv_filename)
digit_train_set = image_info.get_all_info()

# separate training info into samples and target
samples_v1 = digit_train_set[:, 1]
target = digit_train_set[:, 0]
target = target.astype(int)


samples_v2 = list(map(lambda v: np.reshape(v, (-1)), samples_v1))
samples_v3 = image_info.circle_info_arr(samples_v2, samples_v1)




x_train, x_test_before, y_train, y_test = train_test_split(samples_v3,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=10)
