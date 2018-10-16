import numpy as np
import cv2 as cv
import pandas as pd
from image_modifications import ImageModifications
import matplotlib
matplotlib.use("MacOSX")
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# TODO identify if image even has a ship (maybe by pixel count, maybe TF)
# TODO load up large amounts of true data and see what it says: pixels in average ship, how many ships per positive image, shape of average ship, MORE


train_images_filepath = '../../../../image-data-train-test-large-data/airbus-ocean-ship-detection-pictures/train_v2/'
train_image_sub_folder = '0/'
# filename_with = '0a0df8299.jpg'
# filename_with = '0a1a58833.jpg'
# filename_without = '0a00a69de.jpg'  # simple blue ocean
# filename_without = '0a0aeea56.jpg'  # blue ocean with clouds

training_segmentations_filename = '../../resources/ocean-ship-detection/train_ship_segmentations_v2.csv'


segments_df = pd.read_csv(training_segmentations_filename, nrows=100000)
segments_df.fillna('-1', inplace=True)

image_sz = 768
image_mod = ImageModifications(image_sz, segments_df)



import os

bucket_count = 16
assert image_sz % bucket_count == 0
bucket_sz = 49

train_x_list = []
train_y_list = []
test_x_list = []
test_y_list = []

folder_to_examine = train_images_filepath + train_image_sub_folder
for filename in os.listdir(folder_to_examine):
    root, ext = os.path.splitext(filename)
    # if (root.startswith('09') or root.startswith('08') or root.startswith('07') or root.startswith('06') or root.startswith('05')) and ext == '.jpg':
    if (root.startswith('099') or root.startswith('097')) and ext == '.jpg':
        image_to_log = cv.imread(train_images_filepath + train_image_sub_folder + filename)
        image_to_log = cv.blur(image_to_log, (5,5))
        mask_to_log = image_mod.mask_from_filename(filename)
        for x_start in range(0, image_sz, bucket_sz):
            for y_start in range(0, image_sz, bucket_sz):
                x_train = np.array([np.average(image_to_log[x_start:x_start+bucket_sz, y_start:y_start+bucket_sz, 0]),
                                   np.average(image_to_log[x_start:x_start+bucket_sz, y_start:y_start+bucket_sz, 1]),
                                   np.average(image_to_log[x_start:x_start+bucket_sz, y_start:y_start+bucket_sz, 2])], dtype=np.double)
                y_train = np.count_nonzero(mask_to_log[0:bucket_sz, 0:bucket_sz]) > bucket_sz
                if root.startswith('097'):
                    test_x_list.extend([x_train])
                    test_y_list.extend([y_train])
                else:
                    train_x_list.extend([x_train])
                    train_y_list.extend([y_train])


# TODO 1. load up picture
# TODO 2. separate picture into 49 x 49 pixel buckets (16 * 16 buckets per bpicture)
# TODO 3. log pixel info about picture into knn ready info (average r, g, b value), label is if ship is in picture, could also use pixel count instead of T/F for continuous ml options
# TODO 7. Separate into test and train data set
# TODO on test data, do steps 2 & 3, knn (or other) about whether area is t/f

for neighbor_itr in range(3, 4):
    print("neighbors:" + str(neighbor_itr))
    knn = KNeighborsClassifier(n_neighbors=neighbor_itr)
    knn.fit(train_x_list, train_y_list)
    pred = knn.predict(test_x_list)
    y_test_raveled = test_y_list
    score = accuracy_score(y_test_raveled, pred)
    tf_result = y_test_raveled == pred
    print(str(neighbor_itr) + ":" + str(score))



# img_with = cv.imread(train_images_filepath + train_image_sub_folder + filename_with)
# img_without = cv.imread(train_images_filepath + train_image_sub_folder + filename_without)
#
# img_sz = img_with.shape[0]
#
#

#
#
# mask_with = mask_from_filename(filename_with)
# mask_without = mask_from_filename(filename_without)
#
# cv.imshow('without', img_without)
# blur_without = cv.medianBlur(img_without, 5)
# cv.imshow('without_blur', blur_without)
# cv.imshow('mask_without', mask_without)
# cv.imshow('mask_with', mask_with)
# cv.imshow('with', img_with)
# blur_with = cv.blur(img_with, (5, 5))
# blur_with = cv.blur(blur_with, (5, 5))
#
# hue, sat, val = cv.split(cv.cvtColor(blur_with, cv.COLOR_BGR2HSV))
# plt.hist(hue.ravel(),bins=int(180/18),range=[0,181]); plt.show()
# plt.hist(sat.ravel(),bins=8,range=[0,256]); plt.show()
# plt.hist(val.ravel(),bins=8,range=[0,256]); plt.show()
#
# cv.waitKey(0)
# cv.destroyAllWindows()

print("\nfinished")
