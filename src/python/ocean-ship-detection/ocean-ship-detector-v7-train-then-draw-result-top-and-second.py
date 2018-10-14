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
train_image_sub_folder = '3/'
image_path = train_images_filepath + train_image_sub_folder
sample_file_name = '3a0a50b3c.jpg'
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
train_set = pd.read_csv(resources + "train/" + 'jason_top_level_' + str(block_pixels) + '_30.csv', dtype=sub_image_dtype)

x_columns = ['blue_avg', 'green_avg', 'red_avg', 'blue_std', 'green_std', 'red_std']
y_column = ['ship_in_image']
filename_column = ['filename']
train_x_list = train_set[x_columns].values
train_y_list = train_set[y_column].values


review_image = True

filename_start = "3a0"

generate_values = GenerateValues(img_size, block_pixels, 4, False)

columns_to_save = ['ship_in_image', 'blue_avg', 'green_avg', 'red_avg', 'blue_std', 'green_std', 'red_std']

for neighbor_itr in range(3, 4, 1):
    and_count = 0
    bad_guess_count = 0
    non_pred_count = 0
    print("neighbors:" + str(neighbor_itr))
    knn_top = KNeighborsClassifier(n_neighbors=neighbor_itr)
    knn_top.fit(train_x_list, train_y_list.ravel())
    print("training model loaded")

    regex_files = train_images_filepath + train_image_sub_folder + filename_start + "*.jpg"
    images_to_review = glob.glob(regex_files)

    print("images: " + str(len(images_to_review)))
    for itr_idx, filename in enumerate(images_to_review):
        print("\timage: " + str(itr_idx))
        image_to_log = cv.imread(filename)
        image_to_log, thresh_mask = image_mod.adaptive_thresh_mask(image_to_log)
        no_folder_filename = filename.replace(train_images_filepath + train_image_sub_folder, "")
        training_mask = image_mod.mask_from_filename(no_folder_filename)

        no_folder_filename = filename.replace(train_images_filepath + train_image_sub_folder, "")
        image_to_log = cv.imread(filename)
        image_to_log, thresh_mask = image_mod.adaptive_thresh_mask(image_to_log)
        training_mask = image_mod.mask_from_filename(no_folder_filename)
        dict_top_level, dict_second_level = generate_values.parsing_values(image_to_log, training_mask, thresh_mask)

        train_df_top_level = pd.DataFrame.from_dict(dict_top_level, orient='index', columns=columns_to_save)
        train_df_top_level.reset_index(inplace=True)
        train_df_top_level.rename(index=str, columns={"index": "filename"}, inplace=True)

        total_test_x_list = train_df_top_level[x_columns].values
        total_test_y_list = train_df_top_level[y_column].values

        #TODO second level later...large changes probably


        for test_start in range(0, len(total_test_y_list), blocks_in_image):

            test_x_list = total_test_x_list[test_start:test_start + blocks_in_image]
            test_y_list = total_test_y_list[test_start:test_start + blocks_in_image]

            pred = knn_top.predict(test_x_list)

            # expand guesses to neighbors in cardinal directions
            assert pred.shape[0] == blocks * blocks
            pred_update = pred.reshape(-1, blocks)
            pred_update_read = pred_update.copy()
            assert pred_update.shape[0] == blocks
            for x_block in range(blocks):
                for y_block in range(blocks):
                    if pred_update_read[x_block,y_block]:
                        if x_block != 0:
                            pred_update[x_block - 1, y_block] = True
                        if x_block != blocks - 1:
                            pred_update[x_block + 1, y_block] = True
                        if y_block != 0:
                            pred_update[x_block, y_block - 1] = True
                        if y_block != blocks - 1:
                            pred_update[x_block, y_block + 1] = True


            y_test_raveled = test_y_list.ravel()
            tf_result = y_test_raveled == pred

            actual_mask = image_mod.mask_from_filename(filename)
            actual_mask = actual_mask.copy()
            if np.count_nonzero(pred) > 0 or np.count_nonzero(y_test_raveled) > 0:
                this_and_count = np.count_nonzero(np.logical_and(y_test_raveled, pred))
                and_count += this_and_count
                bad_guess_count += (np.count_nonzero(pred) - this_and_count)
                non_pred_count += (np.count_nonzero(y_test_raveled) - this_and_count)

                if review_image:
                    quick_img = cv.imread(filename)
                    blur_and_minimize = image_mod.adaptive_thresh(quick_img)
                    # blur_and_minimize = image_mod.blur_and_minimize(quick_img)


                    for idx, guess in enumerate(pred):
                        actual = y_test_raveled[idx]
                        x_start = (idx % blocks) * block_pixels
                        y_start = int(idx / blocks) * block_pixels
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
                            stop_pt = (x_start + block_pixels, y_start + block_pixels)
                            cv.rectangle(actual_mask, start_pt, stop_pt, brightness, thickness=thickness_to_show)
                            if review_image:
                                cv.rectangle(blur_and_minimize, start_pt, stop_pt, (brightness, brightness, brightness), thickness=thickness_to_show)

                    cv.imshow("actual_mask", actual_mask)
                    cv.imshow('quick', quick_img)
                    cv.imshow('quick_blur_and_minimize', blur_and_minimize)
                    if cv.waitKey(0) & 0xFF == ord('q'):
                        break
    print('\tand_count' + ":" + str(and_count))
    print('\tbad_guess_scount' + ":" + str(bad_guess_count))
    print('\tnon_pred_count' + ":" + str(non_pred_count))



print("\nfinished")
