import numpy as np
import cv2 as cv
import pandas as pd


class ImageModifications:

    def __init__(self, img_sz, segments_df):
        self.img_sz = img_sz
        self.segments_df = segments_df

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
                start_point = int(split[idx]) - 1
                pixels = int(split[idx + 1])
                mask_1d[start_point:start_point+pixels] = 255

    def mask_from_filename(self, filename):
        img_seg_df = self.segments_df.loc[self.segments_df['ImageId'] == filename]
        mask_with = np.zeros((self.img_sz, self.img_sz, 1), dtype=np.uint8)
        self.update_mask_with_segments(mask_with, img_seg_df)
        return np.swapaxes(mask_with, 0, 1)

    # TODO Write this, consider lots of clouds too (though that might not be a concern)
    def blur_and_minimize(self, img):
        img = cv.blur(img, (3,3))
        logical_and = np.ones(img.shape[0:2], dtype=bool)
        for place in range(0, 3):
            color_info = img[:, :, place]
            color_avg = np.average(color_info)
            color_std = np.std(color_info) * 1
            color_std = max((color_std, 20))
            logical_and = np.logical_and(logical_and, color_info > color_avg - color_std)
            logical_and = np.logical_and(logical_and, color_info < color_avg + color_std)
        for place in range(0, 3):
            color_info = img[:, :, place]
            color_info[logical_and] = 0
        img = cv.blur(img, (3, 3))
        return img

    def adaptive_thresh(self, img):
        img, logical_and = self.adaptive_thresh_mask(img)
        for place in range(0, 3):
            color_info = img[:, :, place]
            color_info[logical_and] = 0

        return img

    # TODO update this to remove some of the blur (but after 1st submission)
    def adaptive_thresh_mask(self, img):
        img = cv.blur(img, (5, 5))
        img = cv.blur(img, (5, 5))
        color_images = np.array(cv.split(img))
        for idx, color_img in enumerate(color_images):
            below_adapt = cv.adaptiveThreshold(color_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 101,-10)
            above_adapt = cv.adaptiveThreshold(color_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 101, 10)
            color_images[idx] = np.logical_and(above_adapt, below_adapt)
            color_images[idx] = cv.morphologyEx(color_images[idx], cv.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
            color_images[idx] = cv.erode(color_images[idx], np.ones((3, 3), dtype=np.uint8), iterations=10)
        logical_and = np.ones(img.shape[0:2], dtype=bool)
        for place in range(0, 3):
            logical_and = np.logical_and(logical_and, color_images[place])
        return img, logical_and
