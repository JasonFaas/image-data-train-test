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


# Location of images info
train_images_filepath = '../../../../../image-data-train-test-large-data/airbus-ocean-ship-detection-pictures/train_v2/'
train_image_sub_folder = '3/'
sample_file_name = '3a464ece7.jpg'
# sample_file_name = '3a1631d25.jpg'

# Location of ship locations on images
resources = '../../../resources/ocean-ship-detection/'
training_segmentations_filename = '%strain_ship_segmentations_v2.csv' % resources

segments_df = pd.read_csv(training_segmentations_filename)
segments_df.fillna('-1', inplace=True)

# verify assumptions about image
sample_image = cv.imread(train_images_filepath + train_image_sub_folder + sample_file_name)
image_sz = 768
pixels = (image_sz * image_sz)
assert sample_image.shape[0] == image_sz
image_mod = ImageModifications(image_sz, segments_df)


# list of characters in image filenames
image_filename_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
assert len(image_filename_chars) == 16

filename_start = "3"






# TODO make bucket descriptions different
top_level_bucket_sz = 8
second_bucket_sz = 8

review_images = False
generate_values = GenerateValues(image_sz, top_level_bucket_sz, second_bucket_sz, True, review_images)


folder_filename = train_images_filepath + train_image_sub_folder + sample_file_name
image_to_log = cv.imread(folder_filename)
training_mask = image_mod.mask_from_filename(sample_file_name)
larger_mask = image_mod.increase_area_around_ship(training_mask)


if review_images:
    cv.imshow("image_to_log", image_to_log)
    cv.imshow("training_mask", training_mask)
    cv.imshow("larger_mask", larger_mask)
    cv.waitKey(0)



training_output_filename = resources + "v2/train/jason_top_level_" + str(top_level_bucket_sz) + "_" + filename_start + ".csv"
folder_to_examine = train_images_filepath + train_image_sub_folder



columns_to_save = ['ship_in_image', 'blue_avg', 'green_avg', 'red_avg', 'blue_std', 'green_std', 'red_std']

for idx_1, filename_part in enumerate(image_filename_chars):
    print("\tat " + filename_start + filename_part)
    # go through test data and mark progress
    regex_files = train_images_filepath + train_image_sub_folder + filename_start + filename_part + "*.jpg"
    images_to_review = glob.glob(regex_files)
    if len(images_to_review) == 0:
        print("WARNING: NO FILES FOUND FOR " + regex_files)


    for idx_2, filename in enumerate(images_to_review):
        no_folder_filename = filename.replace(train_images_filepath + train_image_sub_folder, "")

        # acquire images and masks
        image_to_log = cv.imread(filename)
        training_mask = image_mod.mask_from_filename(no_folder_filename)
        if np.count_nonzero(training_mask) == 0:
            print("Skipping " + no_folder_filename)
            continue
        larger_mask = image_mod.increase_area_around_ship(training_mask)


        values_dict = generate_values.parsing_values(image_to_log,
                                                     training_mask,
                                                     larger_mask,
                                                     training=True)

        # log info
        train_df_top_level = pd.DataFrame.from_dict(values_dict, orient='index', columns=columns_to_save)
        train_df_top_level.reset_index(inplace=True)
        train_df_top_level.rename(index=str, columns={"index":"filename"}, inplace=True)

        if idx_1 == 0 and idx_2 == 0:
            train_df_top_level.to_csv(training_output_filename, index=False)
        else:
            train_df_top_level.to_csv(training_output_filename, mode='a', index=False, header=False)


print("\nfinished")
