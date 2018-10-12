import numpy as np
import cv2 as cv
import pandas as pd
from image_modifications import ImageModifications
import matplotlib
matplotlib.use("MacOSX")
from matplotlib import pyplot as plt


train_images_filepath = '../../../../image-data-train-test-large-data/airbus-ocean-ship-detection-pictures/train_v2/'
train_image_sub_folder = '0/'
# filename_with = '0a0df8299.jpg'
filename_with = '002868a5c.jpg'
# filename_without = '0a00a69de.jpg'  # simple blue ocean
filename_without = '0a0aeea56.jpg'  # blue ocean with clouds

training_segmentations_filename = '../../resources/ocean-ship-detection/train_ship_segmentations_v2.csv'


segments_df = pd.read_csv(training_segmentations_filename, nrows=100000)
segments_df.fillna('-1', inplace=True)


img_with = cv.imread(train_images_filepath + train_image_sub_folder + filename_with)
img_without = cv.imread(train_images_filepath + train_image_sub_folder + filename_without)

image_mod = ImageModifications(img_with.shape[0], None)
img_sz = img_with.shape[0]


def mask_from_filename(filename):
    img_seg_df = segments_df.loc[segments_df['ImageId'] == filename]
    mask_with = np.zeros((img_sz, img_sz, 1), dtype=np.uint8)
    image_mod.update_mask_with_segments(mask_with, img_seg_df)
    return np.swapaxes(mask_with, 0, 1)


mask_with = mask_from_filename(filename_with)
mask_without = mask_from_filename(filename_without)

cv.imshow('without', img_without)
blur_without = cv.medianBlur(img_without, 5)
cv.imshow('without_blur', blur_without)
cv.imshow('mask_without', mask_without)
cv.imshow('mask_with', mask_with)
cv.imshow('with', img_with)
blur_with = cv.GaussianBlur(img_with, (5, 5), 5)
hue, sat, val = cv.split(cv.cvtColor(blur_with, cv.COLOR_BGR2HSV))
cv.imshow('with_hue', hue)

plt.hist(hue.ravel(),bins=int(180/18),range=[0,180]); plt.show()
# plt.hist(gre.ravel(),bins=16,range=[0,256]); plt.show()
# plt.hist(red.ravel(),bins=16,range=[0,256]); plt.show()

cv.waitKey(0)
cv.destroyAllWindows()

print("\nfinished")
