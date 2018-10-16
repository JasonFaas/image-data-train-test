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



resources = '../../resources/ocean-ship-detection/'
train_images_filepath = '../../../../image-data-train-test-large-data/airbus-ocean-ship-detection-pictures/train_v2/'
# TODO revert after submission
train_image_sub_folder = '2/'
# train_image_sub_folder = '../test_v2/'
image_path = train_images_filepath + train_image_sub_folder
sample_file_name = '2a0a5eb69.jpg'
test_image = cv.imread(train_images_filepath + train_image_sub_folder + sample_file_name)
training_segmentations_filename = '%strain_ship_segmentations_v2.csv' % resources
segments_df = pd.read_csv(training_segmentations_filename, nrows=100000)
segments_df.fillna('-1', inplace=True)
image_mod = ImageModifications(test_image.shape[0], segments_df)


img_size = test_image.shape[0]
block_pixels = 32
blocks = int(img_size / block_pixels)
blocks_in_image = blocks * blocks

sub_image_dtype = {'filename':str, 'ship_in_image': bool, 'blue_avg': np.double, 'green_avg': np.double, 'red_avg': np.double}
train_set = '30'
top_train_set = pd.read_csv(resources + "train/" + 'jason_top_level_' + str(block_pixels) + '_' + train_set + '.csv', dtype=sub_image_dtype)
second_train_set = pd.read_csv(resources + "train/" + 'jason_second_level_' + str(block_pixels) + '_4_' + train_set + '.csv', dtype=sub_image_dtype)

x_columns = ['blue_avg', 'green_avg', 'red_avg', 'blue_std', 'green_std', 'red_std']
y_column = ['ship_in_image']
top_train_x_list = top_train_set[x_columns].values
top_train_y_list = top_train_set[y_column].values
second_train_x_list = second_train_set[x_columns].values
second_train_y_list = second_train_set[y_column].values


review_image = True

filename_start = "2a2e5aad"

review_warnings = False
generate_values = GenerateValues(img_size, block_pixels, 4, False, review_warnings)

columns_to_save = ['ship_in_image', 'blue_avg', 'green_avg', 'red_avg', 'blue_std', 'green_std', 'red_std']


def update_counts(pred, y_test_raveled, and_count, bad_guess_count, non_pred_count):
    this_and_count = np.count_nonzero(np.logical_and(y_test_raveled, pred))
    and_count += this_and_count
    bad_guess_count += (np.count_nonzero(pred) - this_and_count)
    non_pred_count += (np.count_nonzero(y_test_raveled) - this_and_count)
    return and_count, bad_guess_count, non_pred_count


def result_rectangles(img,
                      predictions,
                      y_test_raveled,
                      pixels_sz,
                      base_x_start,
                      base_y_start,
                      block_count):
    for idx, guess in enumerate(predictions):
        actual = y_test_raveled[idx]
        x_start = (idx % block_count) * pixels_sz + base_x_start
        y_start = int(idx / block_count) * pixels_sz + base_y_start
        if guess or actual:
            brightness = 0
            thickness_to_show = 0
            if guess and actual:
                brightness = 255
                thickness_to_show = 5
            elif actual:
                brightness = 170
                thickness_to_show = 3
            elif guess:
                brightness = 100
                thickness_to_show = 1
            start_pt = (x_start, y_start)
            stop_pt = (x_start + pixels_sz, y_start + pixels_sz)
            if review_image:
                cv.rectangle(img, start_pt, stop_pt, (brightness, brightness, brightness),
                             thickness=thickness_to_show)

def result_mask(predictions,
                pixels_sz,
                base_x_start,
                base_y_start,
                block_count,
                output_mask):
    for idx, guess in enumerate(predictions):
        if guess:
            x_start = (idx % block_count) * pixels_sz + base_x_start
            y_start = int(idx / block_count) * pixels_sz + base_y_start
            brightness = 255
            thickness_to_show = -1
            start_pt = (x_start, y_start)
            stop_pt = (x_start + pixels_sz - 1, y_start + pixels_sz - 1)
            cv.rectangle(output_mask, start_pt, stop_pt, (brightness),thickness=thickness_to_show)

#TODO fix off by one error here too
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


for neighbor_itr in range(3, 4, 1):


    top_and_count = 0
    top_bad_guess_count = 0
    top_non_pred_count = 0
    second_and_count = 0
    second_bad_guess_count = 0
    second_non_pred_count = 0

    top_n_itr = 3
    second_n_itr = neighbor_itr

    print("neighbors:" + str(neighbor_itr))
    knn_top = KNeighborsClassifier(n_neighbors=top_n_itr)
    knn_top.fit(top_train_x_list, top_train_y_list.ravel())
    print("top training model loaded")

    knn_second = KNeighborsClassifier(n_neighbors=second_n_itr)
    knn_second.fit(second_train_x_list, second_train_y_list.ravel())
    print("second training model loaded")

    regex_files = train_images_filepath + train_image_sub_folder + filename_start + "*.jpg"
    images_to_review = glob.glob(regex_files)

    print("images: " + str(len(images_to_review)))
    for itr_idx, filename in enumerate(images_to_review):
        
        no_folder_filename = filename.replace(train_images_filepath + train_image_sub_folder, "")
        print("\timage: " + str(itr_idx) + " " + no_folder_filename)

        image_to_log = cv.imread(filename)
        image_to_log, thresh_mask = image_mod.adaptive_thresh_mask(image_to_log)
        training_mask = image_mod.mask_from_filename(no_folder_filename)

        actual_mask = image_mod.mask_from_filename(no_folder_filename)
        actual_mask = actual_mask.copy()
        output_mask = np.zeros(actual_mask.shape, dtype=np.uint8)

        no_folder_filename = filename.replace(train_images_filepath + train_image_sub_folder, "")
        image_to_log = cv.imread(filename)
        image_to_log, thresh_mask = image_mod.adaptive_thresh_mask(image_to_log)
        training_mask = image_mod.mask_from_filename(no_folder_filename)
        dict_top_level, _ = generate_values.parsing_values(image_to_log, training_mask, thresh_mask)

        train_df_top_level = pd.DataFrame.from_dict(dict_top_level, orient='index', columns=columns_to_save)
        train_df_top_level.reset_index(inplace=True)
        train_df_top_level.rename(index=str, columns={"index": "filename"}, inplace=True)

        top_test_x_list = train_df_top_level[x_columns].values
        top_test_y_list = train_df_top_level[y_column].values

        top_pred = knn_top.predict(top_test_x_list)

        # TODO simplify
        # expand guesses to neighbors in cardinal directions
        assert top_pred.shape[0] == blocks * blocks
        pred_update = top_pred.reshape(-1, blocks)
        pred_update_read = pred_update.copy()
        assert pred_update.shape[0] == blocks
        for x_block in range(blocks):
            for y_block in range(blocks):
                if pred_update_read[x_block,y_block]:
                    for x in range(x_block - 1, x_block + 2):
                        if -1 < x < blocks:
                            for y in range(y_block - 1, y_block + 2):
                                if -1 < y < blocks:
                                    pred_update[x, y] = True

        top_y_test_raveled = top_test_y_list.ravel()


        #TODO second level later...large changes probably

        if review_image:
            quick_img = cv.imread(filename)
            img_draw_on = image_mod.adaptive_thresh(quick_img)
        
        if np.count_nonzero(top_pred) > 0:
            quick_img = cv.imread(filename)
            image_to_log, thresh_mask = image_mod.adaptive_thresh_mask(image_to_log)
            training_mask = image_mod.mask_from_filename(no_folder_filename)

            for idx, guess in enumerate(top_pred):
                if guess:
                    mid_x_start = (idx % blocks) * block_pixels
                    mid_y_start = int(idx / blocks) * block_pixels
                    dict_second_level = generate_values.parse_second_level_values(image_to_log, thresh_mask, training_mask, mid_y_start, mid_y_start+block_pixels, mid_x_start, mid_x_start+block_pixels)

                    train_df_second_level = pd.DataFrame.from_dict(dict_second_level, orient='index', columns=columns_to_save)
                    train_df_second_level.reset_index(inplace=True)
                    train_df_second_level.rename(index=str, columns={"index": "filename"}, inplace=True)

                    second_test_x_list = train_df_second_level[x_columns].values
                    second_test_y_list = train_df_second_level[y_column].values

                    second_pred = knn_second.predict(second_test_x_list)
                    second_y_test_raveled = second_test_y_list.ravel()

                    second_and_count, second_bad_guess_count, second_non_pred_count = update_counts(second_pred,
                                                                                                    second_y_test_raveled,
                                                                                                    second_and_count,
                                                                                                    second_bad_guess_count,
                                                                                                    second_non_pred_count)
                    if review_image:
                        result_rectangles(
                            img_draw_on,
                            second_pred,
                            second_y_test_raveled,
                            4,
                            mid_x_start,
                            mid_y_start,
                            8)
                    result_mask(second_pred,
                                4,
                                mid_x_start,
                                mid_y_start,
                                8,
                                output_mask)



        write_submission_file(output_mask, no_folder_filename, itr_idx)

        if np.count_nonzero(top_pred) > 0 or np.count_nonzero(top_y_test_raveled) > 0:
            # update counts
            top_and_count, top_bad_guess_count, top_non_pred_count = update_counts(top_pred,
                                                                                   top_y_test_raveled,
                                                                                   top_and_count,
                                                                                   top_bad_guess_count,
                                                                                   top_non_pred_count)

            if review_image:
                result_rectangles(img_draw_on, top_pred, top_y_test_raveled, block_pixels, 0, 0, blocks)

                cv.imshow("actual_mask", actual_mask)
                cv.imshow("output_mask", output_mask)
                cv.imshow('quick', quick_img)
                cv.imshow('quick_blur_and_minimize', img_draw_on)
                if cv.waitKey(0) & 0xFF == ord('q'):
                    break
    print('\ttop_and_count' + ":" + str(top_and_count))
    print('\ttop_bad_guess_scount' + ":" + str(top_bad_guess_count))
    print('\ttop_non_pred_count' + ":" + str(top_non_pred_count))
    print('\tsecond_and_count' + ":" + str(second_and_count))
    print('\tsecond_bad_guess_scount' + ":" + str(second_bad_guess_count))
    print('\tsecond_non_pred_count' + ":" + str(second_non_pred_count))



print("\nfinished")
