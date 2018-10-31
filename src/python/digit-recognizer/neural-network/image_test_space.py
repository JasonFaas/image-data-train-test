import numpy as np
import cv2 as cv
import pandas as pd
from sklearn.model_selection import train_test_split


class DisplayImage:

    def get_image_from_values(self, image_values):
        image_1d_array = np.array(image_values)
        image_2d_array = np.reshape(image_1d_array, (-1, 28, 1))
        image_8bit = image_2d_array.astype(np.uint8)
        return image_8bit

    def get_all_info(self, train_file):
        digit_train_set = pd.read_csv(train_file)
        independent_columns = digit_train_set.columns[1:]
        digit_train_set['image'] = digit_train_set.loc[:, independent_columns].values.tolist()

        new_array = np.array(digit_train_set[['label', 'image']].values)
        get_all_images_from_values = list(map(lambda x: self.get_image_from_values(x[1]), new_array))
        new_array[:,1] = get_all_images_from_values
        return new_array

    def train_test_set(self, train_file, train_size, test_file=None, random_state=None):
        # read training info
        digit_train_set = self.get_all_info(train_file)

        # separate training info into samples and target
        samples_v1 = digit_train_set[:, 1]
        target = digit_train_set[:, 0]
        target = target.astype('int')

        x_train = list(map(lambda v: np.reshape(v, (-1)), samples_v1))
        y_train = target
        if test_file is not None and train_size > 0.99:
            digit_test_set = pd.read_csv(test_file)
            x_test = digit_test_set.values
            y_test = None
        else:
            x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                                y_train,
                                                                test_size=1.0 - train_size,
                                                                random_state=random_state)
        return x_train, x_test, y_train, y_test

    def write_to_file(self, predictions_list):
        # Write to file
        result_df = pd.DataFrame(predictions_list)
        result_df.index += (1)
        result_df = result_df.reset_index()
        output_columns = ['ImageId', 'Label']
        result_df.columns = output_columns

        version = '7'
        result_df.to_csv('submission-keras-v%s.csv' % version, index=False)

