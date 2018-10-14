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
train_image_sub_folder = '3/'
sample_file_name = '3a0a50b3c.jpg'

resources = '../../resources/ocean-ship-detection/'
training_segmentations_filename = '%strain_ship_segmentations_v2.csv' % resources

segments_df = pd.read_csv(training_segmentations_filename, nrows=100000)
segments_df.fillna('-1', inplace=True)

sample_image = cv.imread(train_images_filepath + train_image_sub_folder + sample_file_name)
image_sz = 768
assert sample_image.shape[0] == image_sz
image_mod = ImageModifications(image_sz, segments_df)

top_level_bucket_sz = 32
if top_level_bucket_sz == 48:
    top_level_bucket_sz = 48
    top_level_bucket_count = 16
    second_level_bucket_count = 12
    second_level_bucket_sz = 4
elif top_level_bucket_sz == 32:
    top_level_bucket_sz = 32
    top_level_bucket_count = 24
    second_level_bucket_count = 8
    second_level_bucket_sz = 4
elif top_level_bucket_sz == 16:
    top_level_bucket_sz = 16
    top_level_bucket_count = 48
    second_level_bucket_count = 4
    second_level_bucket_sz = 4
else:
    print("failure...")
    exit(1)


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


top_level_output_filename = resources + output_traintest_name + "/jason_top_level_" + str(top_level_bucket_sz) + "_" + filename_start + ".csv"
second_level_output_filename = resources + output_traintest_name + "/jason_second_level_" + str(top_level_bucket_sz) + "_" + str(second_level_bucket_sz) + "_" + filename_start + ".csv"


folder_to_examine = train_images_filepath + train_image_sub_folder
train_dict_second_level = {}


def get_info_to_log(image_to_log, training_mask, thresh_mask):
    image_sz = image_to_log.shape[0]
    pixels = (image_sz * image_sz)
    y_train = np.count_nonzero(training_mask) > pixels * .1
    # TODO make this more efficient
    thresh_approval = pixels - np.count_nonzero(thresh_mask) > pixels * .1

    # logging values
    blue_avg = int(round(np.average(image_to_log[:,:,0])))
    green_avg = int(round(np.average(image_to_log[:,:,1])))
    red_avg = int(round(np.average(image_to_log[:,:,2])))
    blue_std = int(round(np.std(image_to_log[:,:,0])))
    green_std = int(round(np.std(image_to_log[:,:,1])))
    red_std = int(round(np.std(image_to_log[:,:,2])))
    values = (blue_avg, green_avg, red_avg, blue_std, green_std, red_std)

    if y_train and not thresh_approval:
        print("Blurring out valid image!\t" + str(image_sz))


        cv.imshow("itl_slice", image_to_log)
        cv.imshow("training_mask_slice", training_mask)
        thresh_mask = thresh_mask.astype(dtype=np.uint8)
        thresh_mask[:, :] *= 200
        cv.imshow("thresh_mask_slice", thresh_mask)


        if image_sz > 10 and cv.waitKey(10) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            exit(0)
        #values = None
    elif not y_train and not thresh_approval:
        values = (-1, -1, -1, -1, -1, -1)

    return y_train, values

black_images_logged = 0
columns_to_save = ['ship_in_image', 'blue_avg', 'green_avg', 'red_avg', 'blue_std', 'green_std', 'red_std']

for idx, filename_part in enumerate(hex_values):
    print("\tat " + filename_start + filename_part)
    train_dict_top_level = {}
    # go through test data and mark progress
    regex_files = train_images_filepath + train_image_sub_folder + filename_start + filename_part + "*.jpg"
    images_to_review = glob.glob(regex_files)
    if len(images_to_review) == 0:
        print("WARNING: NO FILES FOUND FOR " + regex_files)

    for filename in images_to_review:
        no_folder_filename = filename.replace(train_images_filepath + train_image_sub_folder, "")
        image_to_log = cv.imread(filename)
        image_to_log, thresh_mask = image_mod.adaptive_thresh_mask(image_to_log)
        training_mask = image_mod.mask_from_filename(no_folder_filename)
        for x_top_start in range(0, image_sz, top_level_bucket_sz):
            for y_top_start in range(0, image_sz, top_level_bucket_sz):
                x_top_stop = x_top_start + top_level_bucket_sz
                y_top_stop = y_top_start + top_level_bucket_sz
                itl_slice = image_to_log[x_top_start:x_top_stop, y_top_start:y_top_stop]
                tm_slice = training_mask[x_top_start:x_top_stop, y_top_start:y_top_stop]
                thresh_slice = thresh_mask[x_top_start:x_top_stop, y_top_start:y_top_stop]
                y_train_top, log_values = get_info_to_log(itl_slice, tm_slice, thresh_slice)
                if type(log_values) == type(None):
                    print(str(x_top_start) + " :: " + str(y_top_start))
                    thresh_mask_dsp = thresh_mask.astype(dtype=np.uint8)
                    thresh_mask_dsp[:,:] *= 200

                    cv.imshow("image_to_log", image_to_log)
                    cv.imshow("training_mask", training_mask)
                    cv.imshow("thresh_mask", thresh_mask_dsp)
                    if cv.waitKey(10) & 0xFF == ord('q'):
                        cv.destroyAllWindows()
                        exit(0)
                    else:
                        print("Put more logging in here")
                        continue
                # skip adding info if effectively a black image
                if train_files and log_values[0] == -1 and black_images_logged > 100:
                    continue
                elif log_values[0] == -1:
                    log_values = (0, 0, 0, 0, 0, 0)
                    black_images_logged += 1
                dict_position = no_folder_filename + "_" + str(x_top_start) + "_" + str(y_top_start)
                train_dict_top_level[dict_position] = [y_train_top, log_values[0], log_values[1], log_values[2], log_values[3], log_values[4], log_values[5]]

                # Second level
                for x_second_start in range(x_top_start, x_top_stop, second_level_bucket_sz):
                    for y_second_start in range(y_top_start, x_top_stop, second_level_bucket_sz):
                        x_second_stop = x_second_start + second_level_bucket_sz
                        y_second_stop = y_second_start + second_level_bucket_sz
                        itl_slice = image_to_log[x_second_start:x_second_stop, y_second_start:y_second_stop]
                        tm_slice = training_mask[x_second_start:x_second_stop, y_second_start:y_second_stop]
                        thresh_slice = thresh_mask[x_second_start:x_second_stop, y_second_start:y_second_stop]
                        y_train_second, log_values = get_info_to_log(itl_slice, tm_slice, thresh_slice)

                        if y_train_top or y_train_second:
                            if type(log_values) == type(None):
                                print("fix second level")
                                continue

                            train_dict_second_level[no_folder_filename + "_" + str(x_second_start) + "_" + str(y_second_start)] = [y_train_second, log_values[0], log_values[1], log_values[2], log_values[3], log_values[4], log_values[5]]
    train_df_top_level = pd.DataFrame.from_dict(train_dict_top_level, orient='index', columns=columns_to_save)
    train_df_top_level.reset_index(inplace=True)
    train_df_top_level.rename(index=str, columns={"index":"filename"}, inplace=True)
    if idx == 0:
        train_df_top_level.to_csv(top_level_output_filename, index=False)
    else:
        train_df_top_level.to_csv(top_level_output_filename, mode='a', index=False, header=False)


train_df_second_level = pd.DataFrame.from_dict(train_dict_second_level, orient='index', columns=columns_to_save)
train_df_second_level.reset_index(inplace=True)
train_df_second_level.rename(index=str, columns={"index":"filename"}, inplace=True)
train_df_second_level.to_csv(second_level_output_filename, index=False)

print("\nfinished")
