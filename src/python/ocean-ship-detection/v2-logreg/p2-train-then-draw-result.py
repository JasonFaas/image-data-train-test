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
from generate_values import GenerateValues


# Base resources setup
resources = '../../../resources/ocean-ship-detection/'
test_images_filepath = '../../../../../image-data-train-test-large-data/airbus-ocean-ship-detection-pictures/train_v2/'

# Test images location
test_image_sub_folder = '2/'
# test_image_sub_folder = '../test_v2/'
filename_start = "2bbb"
sample_file_name = '2a0dc6a3a.jpg'

# Segments for verification
training_segmentations_filename = '%strain_ship_segmentations_v2.csv' % resources
segments_df = pd.read_csv(training_segmentations_filename, nrows=100000)

# Base info sample image image
test_image = cv.imread(test_images_filepath + test_image_sub_folder + sample_file_name)
img_size = test_image.shape[0]

segments_df.fillna('-1', inplace=True)
image_mod = ImageModifications(test_image.shape[0], segments_df)
block_pixels = 8
blocks = int(img_size / block_pixels)
blocks_in_image = blocks * blocks

# Read training data compressed to CSV
train_set = '3a'
sub_image_dtype = {'filename':str, 'ship_in_image': bool, 'blue_avg': np.double, 'green_avg': np.double, 'red_avg': np.double}
top_train_set = pd.read_csv(resources + "v2/train/" + 'jason_top_level_' + str(block_pixels) + '_' + train_set + '.csv', dtype=sub_image_dtype)

x_columns = ['blue_avg', 'green_avg', 'red_avg', 'blue_std', 'green_std', 'red_std']
y_column = ['ship_in_image']
top_train_x_list = top_train_set[x_columns].values
top_train_y_list = top_train_set[y_column].values

review_image = True

generate_values = GenerateValues(img_size, block_pixels, 4, False)

columns_to_save = ['ship_in_image', 'blue_avg', 'green_avg', 'red_avg', 'blue_std', 'green_std', 'red_std']


def update_counts(pred, y_test_raveled, and_count, bad_guess_count, non_pred_count):
    this_and_count = np.count_nonzero(np.logical_and(y_test_raveled, pred))
    and_count += this_and_count
    bad_guess_count += (np.count_nonzero(pred) - this_and_count)
    non_pred_count += (np.count_nonzero(y_test_raveled) - this_and_count)
    return and_count, bad_guess_count, non_pred_count




def write_submission_file(output_mask, no_folder_filename, itr_idx):
    # create data frame
    swapped_mask = np.swapaxes(output_mask, 0, 1)
    single_row_mask = swapped_mask.reshape(-1)
    string_output = ''
    current_length = 0
    for srm_idx, val in enumerate(single_row_mask):
        if current_length == 0 and val > 0:
            if string_output == '':
                string_output = str(srm_idx + 1)
            else:
                string_output = string_output + " " + str(srm_idx + 1)
            current_length += 1
        elif val > 0:
            current_length += 1
        elif current_length != 0 and val == 0:
            string_output = string_output + " " + str(current_length)
            current_length = 0
    if current_length != 0:
        string_output = string_output + " " + str(current_length)

    result_df = pd.DataFrame(data=[[no_folder_filename, string_output]], columns=['ImageId','EncodedPixels'])

    # write data to file
    csv = 'jason_submission_updated_by_one.csv'
    if itr_idx == 0:
        result_df.to_csv(resources + csv, index=False)
    else:
        result_df.to_csv(resources + csv, mode='a', index=False, header=False)




top_and_count = 0
top_bad_guess_count = 0
top_non_pred_count = 0
second_and_count = 0
second_bad_guess_count = 0
second_non_pred_count = 0


print("loading data into LogisticRegression:")
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(top_train_x_list, top_train_y_list)
print("training model loaded")

regex_files = test_images_filepath + test_image_sub_folder + filename_start + "*.jpg"
images_to_review = glob.glob(regex_files)

print("images: " + str(len(images_to_review)))
for itr_idx, filename in enumerate(images_to_review):

    no_folder_filename = filename.replace(test_images_filepath + test_image_sub_folder, "")
    print("\timage: " + str(itr_idx) + " " + no_folder_filename)

    # acquire images and masks
    image_to_log = cv.imread(filename)
    training_mask = image_mod.mask_from_filename(no_folder_filename)
    larger_mask = image_mod.increase_area_around_ship(training_mask)

    values_dict = generate_values.parsing_values(image_to_log, training_mask, larger_mask, training=False)


    train_df_top_level = pd.DataFrame.from_dict(values_dict, orient='index', columns=columns_to_save)
    train_df_top_level.reset_index(inplace=True)
    train_df_top_level.rename(index=str, columns={"index": "filename"}, inplace=True)

    top_test_x_list = train_df_top_level[x_columns].values
    top_test_y_list = train_df_top_level[y_column].values

    top_pred = model.predict(top_test_x_list)

    # TODO simplify
    # remove guess if no neighbors
    pred_update = top_pred.reshape(-1, blocks)
    for x_block in range(1, blocks - 1):
        for y_block in range(1, blocks - 1):
            if pred_update[x_block,y_block]:
                pred_update[x_block,y_block] = pred_update[x_block - 1, y_block] \
                                               or pred_update[x_block + 1, y_block] \
                                               or pred_update[x_block, y_block - 1] \
                                               or pred_update[x_block, y_block + 1]

    # TODO simplify
    # expand guesses to neighbors in 8 directions
    pred_update = top_pred.reshape(-1, blocks)
    pred_update_read = pred_update.copy()
    for x_block in range(blocks):
        for y_block in range(blocks):
            if pred_update_read[x_block,y_block]:
                for x in range(x_block - 1, x_block + 2):
                    if -1 < x < blocks:
                        for y in range(y_block - 1, y_block + 2):
                            if -1 < y < blocks:
                                pred_update[x, y] = True

    top_y_test_raveled = top_test_y_list.ravel()

    output_mask = np.zeros(training_mask.shape, dtype=np.uint8)
    write_submission_file(output_mask, no_folder_filename, itr_idx)

    if np.count_nonzero(top_pred) > 0 or np.count_nonzero(top_y_test_raveled) > 0:
        # update counts
        top_and_count, top_bad_guess_count, top_non_pred_count = update_counts(top_pred,
                                                                               top_y_test_raveled,
                                                                               top_and_count,
                                                                               top_bad_guess_count,
                                                                               top_non_pred_count)

        if review_image:
            image_mod.result_rectangles(image_to_log, top_pred, top_y_test_raveled, block_pixels, 0, 0, blocks)
            cv.imshow('image_to_log', image_to_log)
            if cv.waitKey(0) & 0xFF == ord('q'):
                break

print('\ttop_and_count' + ":" + str(top_and_count))
print('\ttop_bad_guess_count' + ":" + str(top_bad_guess_count))
print('\ttop_non_pred_count' + ":" + str(top_non_pred_count))


print("\nfinished")
