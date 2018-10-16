import numpy as np
import cv2 as cv
import pandas as pd
from image_modifications import ImageModifications
import matplotlib
matplotlib.use("MacOSX")
from matplotlib import pyplot as plt
import glob
from image_modifications import ImageModifications
from generate_values import GenerateValues


train_images_filepath = '../../../../image-data-train-test-large-data/airbus-ocean-ship-detection-pictures/train_v2/'
train_image_sub_folder = '3/'
sample_file_name = '3a0a50b3c.jpg'

resources = '../../resources/ocean-ship-detection/'
training_segmentations_filename = '%strain_ship_segmentations_v2.csv' % resources

segments_df = pd.read_csv(training_segmentations_filename)
segments_df.fillna('-1', inplace=True)

sample_image = cv.imread(train_images_filepath + train_image_sub_folder + sample_file_name)
image_sz = 768
pixels = (image_sz * image_sz)
assert sample_image.shape[0] == image_sz
image_mod = ImageModifications(image_sz, segments_df)
top_level_bucket_sz = 32
second_bucket_size = 4

review_warnings = False
generate_values = GenerateValues(image_sz, top_level_bucket_sz, second_bucket_size, True, review_warnings)


hex_values = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
assert len(hex_values) == 16

filename_start = "3"


top_level_output_filename = resources + "train/jason_top_level_" + str(top_level_bucket_sz) + "_" + filename_start + ".csv"
second_level_output_filename = resources + "train/jason_second_level_" + str(top_level_bucket_sz) + "_" + str(second_bucket_size) + "_" + filename_start + ".csv"


folder_to_examine = train_images_filepath + train_image_sub_folder



columns_to_save = ['ship_in_image', 'blue_avg', 'green_avg', 'red_avg', 'blue_std', 'green_std', 'red_std']

for idx_1, filename_part in enumerate(hex_values):
    print("\tat " + filename_start + filename_part)
    # go through test data and mark progress
    regex_files = train_images_filepath + train_image_sub_folder + filename_start + filename_part + "*.jpg"
    images_to_review = glob.glob(regex_files)
    if len(images_to_review) == 0:
        print("WARNING: NO FILES FOUND FOR " + regex_files)

    for idx_2, filename in enumerate(images_to_review):
        no_folder_filename = filename.replace(train_images_filepath + train_image_sub_folder, "")
        image_to_log = cv.imread(filename)
        image_to_log, thresh_mask = image_mod.adaptive_thresh_mask(image_to_log)
        training_mask = image_mod.mask_from_filename(no_folder_filename)
        dict_top_level, dict_second_level = generate_values.parsing_values(image_to_log, training_mask, thresh_mask)

        # log info
        train_df_top_level = pd.DataFrame.from_dict(dict_top_level, orient='index', columns=columns_to_save)
        train_df_top_level.reset_index(inplace=True)
        train_df_top_level.rename(index=str, columns={"index":"filename"}, inplace=True)

        train_df_second_level = pd.DataFrame.from_dict(dict_second_level, orient='index', columns=columns_to_save)
        train_df_second_level.reset_index(inplace=True)
        train_df_second_level.rename(index=str, columns={"index": "filename"}, inplace=True)

        if idx_1 == 0 and idx_2 == 0:
            train_df_top_level.to_csv(top_level_output_filename, index=False)
            train_df_second_level.to_csv(second_level_output_filename, index=False)

        else:
            train_df_top_level.to_csv(top_level_output_filename, mode='a', index=False, header=False)
            train_df_second_level.to_csv(second_level_output_filename, mode='a', index=False, header=False)


print("\nfinished")
