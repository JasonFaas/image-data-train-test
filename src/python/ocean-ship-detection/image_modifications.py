import numpy as np
import cv2 as cv
import pandas as pd


class ImageModifications:

    def __init__(self, img_sz):
        self.img_sz = img_sz

    @staticmethod
    def get_image_from_values(self, image_values):
        return np.reshape(image_values, (-1, self.img_sz, 1))

    def values_from_img(self, np_image):
        return np.reshape(np_image, (-1, 1))

    def update_mask_with_segments(self, mask_2d, df_segments):
        mask_1d = self.values_from_img(mask_2d)
        for row in df_segments.values:
            split = row[1].split(' ')
            if len(split) < 2:
                continue
            for idx in range(0, len(split), 2):
                start_point = int(split[idx])
                pixels = int(split[idx + 1])
                mask_1d[start_point:start_point+pixels] = 255

