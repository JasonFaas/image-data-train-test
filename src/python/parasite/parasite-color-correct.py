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




images_to_review = glob.glob(large_resources + "0*5" + ".jpg")
# xml_to_review = glob.glob(small_resources + "*" + ".xml")


for idx, img_filename in enumerate(images_to_review):
    base_name = img_filename[-8:-4]
    print("\n" + base_name)
    xml_filename = small_resources + base_name + ".xml"

    img = cv.imread(img_filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blu, gre, red = cv.split(img)


    blu_median = np.median(blu)
    gre_median = np.median(gre)
    red_median = np.median(red)


    print(red_median)
    print(gre_median)
    print(blu_median)

    # new_image = np.zeros(img.shape, img.dtype)
    invGamma = 1.0 / 1.5
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    new_image = cv.LUT(img, table)

    cv.imshow("new", new_image)
    cv.imshow("img", img)
    cv.waitKey(0)
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
