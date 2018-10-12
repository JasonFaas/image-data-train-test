import numpy as np
import cv2 as cv
import pandas as pd
from image_modifications import ImageModifications
import matplotlib
matplotlib.use("MacOSX")
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import glob


train_images_filepath = '../../../../image-data-train-test-large-data/airbus-ocean-ship-detection-pictures/train_v2/'
train_image_sub_folder = '0/'
sample_file_name = '0a0df8299.jpg'

resources = '../../resources/ocean-ship-detection/'
training_segmentations_filename = '%strain_ship_segmentations_v2.csv' % resources

segments_df = pd.read_csv(training_segmentations_filename, nrows=100000)
segments_df.fillna('-1', inplace=True)

sample_image = cv.imread(train_images_filepath + train_image_sub_folder + sample_file_name)
image_sz = 768
assert sample_image.shape[0] == image_sz
image_mod = ImageModifications(image_sz, segments_df)

bucket_count = 12
bucket_sz = 64
assert image_sz % bucket_count == 0
assert int(image_sz / bucket_count) == bucket_sz

hex_values = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
assert len(hex_values) == 16

filename_start = "09"
output_filename = resources + "jason_top_level_" + str(bucket_sz) + "_" + filename_start + ".csv"

folder_to_examine = train_images_filepath + train_image_sub_folder
for idx, filename_part in enumerate(hex_values):
    print("\tat " + filename_start + filename_part)
    train_dict_top_level = {}
    # go through test data and mark progress
    regex_files = train_images_filepath + train_image_sub_folder + filename_start + filename_part + "*.jpg"
    images_to_review = glob.glob(regex_files)
    if len(images_to_review) == 0:
        print("WARNING: NO FILES FOUND FOR " + regex_files)

    for filename in images_to_review:
        image_to_log = cv.imread(filename)
        no_folder_filename = filename.replace(train_images_filepath + train_image_sub_folder, "")
        mask_to_log = image_mod.mask_from_filename(no_folder_filename)
        for x_start in range(0, image_sz, bucket_sz):
            for y_start in range(0, image_sz, bucket_sz):
                # TODO consider skip every other non-ship frame if on same line (basically cut data in half from fewer 'falses')
                blue_avg = np.average(image_to_log[x_start:x_start+bucket_sz, y_start:y_start+bucket_sz, 0])
                green_avg = np.average(image_to_log[x_start:x_start+bucket_sz, y_start:y_start+bucket_sz, 1])
                red_avg = np.average(image_to_log[x_start:x_start+bucket_sz, y_start:y_start+bucket_sz, 2])
                y_train = np.count_nonzero(mask_to_log[x_start:x_start+bucket_sz, y_start:y_start+bucket_sz]) > bucket_sz
                train_dict_top_level[no_folder_filename + "_" + str(x_start) + "_" + str(y_start)] = [y_train, blue_avg, green_avg, red_avg]
    train_df_top_level = pd.DataFrame.from_dict(train_dict_top_level, orient='index', columns=['ship_in_image', 'blue_avg', 'green_avg', 'red_avg'])
    if idx == 0:
        train_df_top_level.to_csv(output_filename, index=False)
    else:
        train_df_top_level.to_csv(output_filename, mode='a', index=False, header=False)


print("\nfinished")
