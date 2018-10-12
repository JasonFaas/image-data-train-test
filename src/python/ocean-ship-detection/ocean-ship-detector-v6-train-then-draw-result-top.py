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



sub_image_dtype = {'filename':str, 'ship_in_image': bool, 'blue_avg': np.double, 'green_avg': np.double, 'red_avg': np.double}
train_set = pd.read_csv(resources + 'jason_top_level_64_08.csv', dtype=sub_image_dtype)
test_set = pd.read_csv(resources + 'jason_top_level_64_070.csv', dtype=sub_image_dtype)

x_columns = ['blue_avg', 'green_avg', 'red_avg']
y_column = ['ship_in_image']
filename_column = ['filename']
train_x_list = train_set[x_columns].values
train_y_list = train_set[y_column].values
total_test_file_list = test_set[filename_column].values
total_test_x_list = test_set[x_columns].values
total_test_y_list = test_set[y_column].values


# TODO unhardcode this
blocks = 12
block_pixels = 64
blocks_in_image = blocks * blocks

for neighbor_itr in range(3, 4):
    print("neighbors:" + str(neighbor_itr))
    knn = KNeighborsClassifier(n_neighbors=neighbor_itr)
    knn.fit(train_x_list, train_y_list.ravel())
    for test_start in range(0, len(total_test_y_list), blocks_in_image):

        filename = str(total_test_file_list[test_start])
        filename = filename[2:filename.find("_")]
        print("Filename here:" + filename)
        test_x_list = total_test_x_list[test_start:test_start + blocks_in_image]
        test_y_list = total_test_y_list[test_start:test_start + blocks_in_image]

        pred = knn.predict(test_x_list)
        y_test_raveled = test_y_list.ravel()
        tf_result = y_test_raveled == pred

        # TODO remove break after 1st success
        actual_mask = image_mod.mask_from_filename(filename)
        actual_mask = actual_mask.copy()
        if np.count_nonzero(pred) > 0 or np.count_nonzero(y_test_raveled) > 0:
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
                    print(filename)
                    cv.rectangle(actual_mask, start_pt, stop_pt, brightness, thickness=thickness_to_show)
            cv.imshow("actual_mask", actual_mask)
            if cv.waitKey(0) & 0xFF == ord('q'):
                break





print("\nfinished")
