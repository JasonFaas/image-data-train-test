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

    def increase_area_around_ship(self, ship_mask):
        return cv.dilate(ship_mask, np.ones((9, 9), dtype=np.uint8), iterations=9)

    # Add rectangles to image for reviewing
    def result_rectangles(self,
                          img,
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
                brightness = 100
                thickness_to_show = 1
                if guess and actual:
                    brightness = 255
                    thickness_to_show = 5
                elif guess:
                    brightness = 170
                    thickness_to_show = 3
                start_pt = (x_start, y_start)
                stop_pt = (x_start + pixels_sz, y_start + pixels_sz)
                cv.rectangle(img,
                             start_pt,
                             stop_pt,
                             (brightness, brightness, brightness),
                             thickness=thickness_to_show)


    def result_mask(self,
                    predictions,
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