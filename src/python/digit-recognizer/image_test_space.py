import numpy as np
import cv2 as cv
import pandas as pd

class DisplayImage:

    def __init__(self, file_path=None, data_set=None):
        if type(file_path) != type(None):
            self.digit_train_set = pd.read_csv(file_path)
            independent_columns = self.digit_train_set.columns[1:]
            self.digit_train_set['image'] = self.digit_train_set.loc[:, independent_columns].values.tolist()
        elif type(data_set) != type(None):
            self.digit_train_set = pd.DataFrame({'image':data_set.tolist()})

    def display_image(self, location=None):
        image_8bit = self.get_image_from_matrix(location)

        cv.imshow("img_2", image_8bit)
        return cv.waitKey(0) & 0xFF
        # cv.destroyAllWindows()

    def get_image_from_matrix(self, location):
        image_values = self.digit_train_set.loc[location, ['image']].values
        image_8bit = self.get_image_from_values(image_values[0])
        return image_8bit

    def get_image_from_values(self, image_values):
        image_1d_array = np.array(image_values)
        image_2d_array = np.reshape(image_1d_array, (-1, 28, 1))
        image_8bit = image_2d_array.astype(np.uint8)
        return image_8bit

    def get_all_info(self):
        new_array = np.array(self.digit_train_set[['label', 'image']].values)
        get_all_images_from_values = list(map(lambda x: self.get_image_from_values(x[1]), new_array))
        new_array[:,1] = get_all_images_from_values
        return new_array
