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

small_resources = '../../resources/parasite/label/'
large_resources = '../../../../image-data-train-test-large-data/Coccidia/img/'



# train_set = pd.read_csv(csv_filename)
# print(train_set.columns)
# first_row = train_set.loc[0]
# print(first_row)
# print(train_set.iloc(0).describe())
# print(train_set.iloc(0).head())

# image = first_row['drawing']


# import xml.etree.ElementTree as ET
# root = ET.parse(xml_filename).getroot()
# for elem in root:
#     for subelem in elem.findall('')
#     print(elem.find('xmin'))

# img_filename = large_resources + train_file + ".jpg"

images_to_review = glob.glob(large_resources + "*" + ".jpg")
# xml_to_review = glob.glob(small_resources + "*" + ".xml")
largest_side = 0

for idx, img_filename in enumerate(images_to_review):
    base_name = img_filename[-8:-4]
    print(base_name)
    xml_filename = small_resources + base_name + ".xml"

    img = cv.imread(img_filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blu, gre, red = cv.split(img)

    from xml.dom import minidom
    mydoc = minidom.parse(xml_filename)

    def get_single_data_from_xml(tag, idx):
        return int(mydoc.getElementsByTagName(tag)[idx].firstChild.data)

    import random
    while True:
        block = random.randint(150,201) * 2 + 1
        C = random.randint(55, 60)
        print("\n")
        print(block)
        print(C)

        gaus = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, block, C)
        for rect_idx in range(len(mydoc.getElementsByTagName('xmin'))):
            xmin = get_single_data_from_xml('xmin', rect_idx)
            ymin = get_single_data_from_xml('ymin', rect_idx)
            xmax = get_single_data_from_xml('xmax', rect_idx)
            ymax = get_single_data_from_xml('ymax', rect_idx)
            cv.rectangle(gray, (xmin, ymin), (xmax, ymax), (255))
            cv.rectangle(red, (xmin, ymin), (xmax, ymax), (255))
            cv.rectangle(gre, (xmin, ymin), (xmax, ymax), (255))
            cv.rectangle(blu, (xmin, ymin), (xmax, ymax), (255))
            cv.rectangle(gaus, (xmin, ymin), (xmax, ymax), (255))
            if xmax - xmin > largest_side:
                largest_side = xmax - xmin
            if ymax - ymin > largest_side:
                largest_side = ymax - ymin
        print(img.shape)



        cv.imshow("zeros", img)
        cv.imshow("gray", gray)
        cv.imshow("red", red)
        cv.imshow("gre", gre)
        cv.imshow("blu", blu)
        cv.imshow("gaus", gaus)
        key = cv.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            break
    if key == ord('q'):
        break


print(largest_side)
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
