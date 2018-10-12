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
train_image_sub_folder = '1/'
sample_file_name = '1a0a0f60d.jpg'

resources = '../../resources/ocean-ship-detection/'
training_segmentations_filename = '%strain_ship_segmentations_v2.csv' % resources

segments_df = pd.read_csv(training_segmentations_filename, nrows=100000)
segments_df.fillna('-1', inplace=True)

sample_image = cv.imread(train_images_filepath + train_image_sub_folder + sample_file_name)
image_sz = 768
assert sample_image.shape[0] == image_sz
image_mod = ImageModifications(image_sz, segments_df)

larger_bucket = True
if larger_bucket:
    top_level_bucket_count = 16
    top_level_bucket_sz = 48
    second_level_bucket_count = 12
    second_level_bucket_sz = 4
else:
    top_level_bucket_count = 24
    top_level_bucket_sz = 32
    second_level_bucket_count = 8
    second_level_bucket_sz = 4

assert image_sz % top_level_bucket_count == 0
assert int(image_sz / top_level_bucket_count) == top_level_bucket_sz

assert top_level_bucket_sz % second_level_bucket_count == 0
assert int(top_level_bucket_sz / second_level_bucket_count) == second_level_bucket_sz

hex_values = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
assert len(hex_values) == 16

train_files = True
output_traintest_name = "test"
if train_files:
    output_traintest_name = "train"
filename_start = "1"


top_level_output_filename = resources + "jason_top_level_" + output_traintest_name + "_" + str(top_level_bucket_sz) + "_" + filename_start + ".csv"
second_level_output_filename = resources + "jason_second_level_" + output_traintest_name + "_" + str(top_level_bucket_sz) + "_" + str(second_level_bucket_sz) + "_" + filename_start + ".csv"


folder_to_examine = train_images_filepath + train_image_sub_folder
train_dict_second_level = {}
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
        # TODO cleanup this REPETITIVE code
        for x_top_start in range(0, image_sz, top_level_bucket_sz):
            prev_y_train = True
            for y_top_start in range(0, image_sz, top_level_bucket_sz):
                y_train = np.count_nonzero(mask_to_log[x_top_start:x_top_start + top_level_bucket_sz, y_top_start:y_top_start + top_level_bucket_sz]) > top_level_bucket_sz
                # skip adding info if previous and current and next are all false
                if train_files and not prev_y_train and not y_train and y_top_start + top_level_bucket_sz != image_sz:
                    next_y_train = np.count_nonzero(mask_to_log[x_top_start:x_top_start + top_level_bucket_sz, y_top_start + top_level_bucket_sz:y_top_start + top_level_bucket_sz*2]) > top_level_bucket_sz
                    if not next_y_train:
                        prev_y_train = True
                        continue
                blue_avg = np.average(image_to_log[x_top_start:x_top_start + top_level_bucket_sz, y_top_start:y_top_start + top_level_bucket_sz, 0])
                green_avg = np.average(image_to_log[x_top_start:x_top_start + top_level_bucket_sz, y_top_start:y_top_start + top_level_bucket_sz, 1])
                red_avg = np.average(image_to_log[x_top_start:x_top_start + top_level_bucket_sz, y_top_start:y_top_start + top_level_bucket_sz, 2])
                train_dict_top_level[no_folder_filename + "_" + str(x_top_start) + "_" + str(y_top_start)] = [y_train, blue_avg, green_avg, red_avg]
                prev_y_train = y_train
                if y_train:
                    for x_second_start in range(x_top_start, x_top_start + top_level_bucket_sz, second_level_bucket_sz):
                        for y_second_start in range(y_top_start, y_top_start + top_level_bucket_sz, second_level_bucket_sz):
                            y_train = np.count_nonzero(mask_to_log[x_second_start:x_second_start + second_level_bucket_sz,y_second_start:y_second_start + second_level_bucket_sz]) > second_level_bucket_sz
                            blue_avg = np.average(image_to_log[x_second_start:x_second_start + second_level_bucket_sz,y_second_start:y_second_start + second_level_bucket_sz, 0])
                            green_avg = np.average(image_to_log[x_second_start:x_second_start + second_level_bucket_sz,y_second_start:y_second_start + second_level_bucket_sz, 1])
                            red_avg = np.average(image_to_log[x_second_start:x_second_start + second_level_bucket_sz,y_second_start:y_second_start + second_level_bucket_sz, 2])
                            train_dict_second_level[no_folder_filename + "_" + str(x_second_start) + "_" + str(y_second_start)] = [y_train, blue_avg, green_avg, red_avg]
    train_df_top_level = pd.DataFrame.from_dict(train_dict_top_level, orient='index', columns=['ship_in_image', 'blue_avg', 'green_avg', 'red_avg'])
    train_df_top_level.reset_index(inplace=True)
    train_df_top_level.rename(index=str, columns={"index":"filename"}, inplace=True)
    if idx == 0:
        train_df_top_level.to_csv(top_level_output_filename, index=False)
    else:
        train_df_top_level.to_csv(top_level_output_filename, mode='a', index=False, header=False)

train_df_second_level = pd.DataFrame.from_dict(train_dict_second_level, orient='index', columns=['ship_in_image', 'blue_avg', 'green_avg', 'red_avg'])
train_df_second_level.reset_index(inplace=True)
train_df_second_level.rename(index=str, columns={"index":"filename"}, inplace=True)
train_df_second_level.to_csv(second_level_output_filename, index=False)

print("\nfinished")
