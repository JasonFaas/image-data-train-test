import numpy as np
import cv2 as cv
import pandas as pd
from image_modifications import ImageModifications
import matplotlib
matplotlib.use("MacOSX")
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



resources = '../../resources/ocean-ship-detection/'
train_images_filepath = '../../../../image-data-train-test-large-data/airbus-ocean-ship-detection-pictures/train_v2/'
train_image_sub_folder = '0/'
image_path = train_images_filepath + train_image_sub_folder
sample_file_name = '0a0df8299.jpg'
test_image = cv.imread(train_images_filepath + train_image_sub_folder + sample_file_name)
training_segmentations_filename = '%strain_ship_segmentations_v2.csv' % resources
segments_df = pd.read_csv(training_segmentations_filename, nrows=100000)
segments_df.fillna('-1', inplace=True)
image_mod = ImageModifications(test_image.shape[0], segments_df)


img_size = test_image.shape[0]
block_pixels = 48
blocks = int(img_size / block_pixels)
blocks_in_image = blocks * blocks

sub_image_dtype = {'filename':str, 'ship_in_image': bool, 'blue_avg': np.double, 'green_avg': np.double, 'red_avg': np.double}
train_set = pd.read_csv(resources + 'jason_top_level_train_' + str(block_pixels) + '_08.csv', dtype=sub_image_dtype)
test_set = pd.read_csv(resources + 'jason_top_level_test_' + str(block_pixels) + '_080.csv', dtype=sub_image_dtype)

x_columns = ['blue_avg', 'green_avg', 'red_avg']
y_column = ['ship_in_image']
filename_column = ['filename']
train_x_list = train_set[x_columns].values
train_y_list = train_set[y_column].values
total_test_file_list = test_set[filename_column].values
total_test_x_list = test_set[x_columns].values
total_test_y_list = test_set[y_column].values


for neighbor_itr in range(4, 5):
    and_count = 0
    bad_guess_count = 0
    non_pred_count = 0
    print("neighbors:" + str(neighbor_itr))
    knn = KNeighborsClassifier(n_neighbors=neighbor_itr)
    knn.fit(train_x_list, train_y_list.ravel())
    for test_start in range(0, len(total_test_y_list), blocks_in_image):

        filename = str(total_test_file_list[test_start])
        filename = filename[2:filename.find("_")]
        test_x_list = total_test_x_list[test_start:test_start + blocks_in_image]
        test_y_list = total_test_y_list[test_start:test_start + blocks_in_image]

        pred = knn.predict(test_x_list)
        y_test_raveled = test_y_list.ravel()
        tf_result = y_test_raveled == pred

        actual_mask = image_mod.mask_from_filename(filename)
        actual_mask = actual_mask.copy()
        if np.count_nonzero(pred) > 0 or np.count_nonzero(y_test_raveled) > 0:
            this_and_count = np.count_nonzero(np.logical_and(y_test_raveled, pred))
            and_count += this_and_count
            bad_guess_count += (np.count_nonzero(pred) - this_and_count)
            non_pred_count += (np.count_nonzero(y_test_raveled) - this_and_count)

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
                    elif guess:
                        brightness = 170
                        thickness_to_show = 3
                    elif actual:
                        brightness = 100
                        thickness_to_show = 1
                    start_pt = (x_start, y_start)
                    stop_pt = (x_start + block_pixels, y_start + block_pixels)
                    cv.rectangle(actual_mask, start_pt, stop_pt, brightness, thickness=thickness_to_show)
            cv.imshow("actual_mask", actual_mask)
            quick_img = cv.imread(train_images_filepath + train_image_sub_folder + filename)
            cv.imshow('quick', quick_img)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
    print('\tand_count' + ":" + str(and_count))
    print('\tbad_guess_scount' + ":" + str(bad_guess_count))
    print('\tnon_pred_count' + ":" + str(non_pred_count))



print("\nfinished")
