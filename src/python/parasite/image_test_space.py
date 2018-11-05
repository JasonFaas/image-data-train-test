import numpy as np
import cv2 as cv
import pandas as pd
from xml.dom import minidom


class DisplayImage:
    
    def __init__(self, img_size, screen_size):
        self.img_size = img_size
        self.screen_size = screen_size
        self.small_resources = '../../resources/parasite/label/'
        self.display_image_creation = False


    def two_points_distance(self, pt_a, pt_b):
        return ((pt_a[0] - pt_b[0]) ** 2 + (pt_a[1] - pt_b[1]) ** 2) ** 0.5

    def get_roi(self, xmin_org, xmax_org, ymin_org, ymax_org, pad, size, param):
        if param[0] == 't':
            xmin = xmin_org - pad
            xmax = xmin + size
        else:
            xmax = xmax_org + pad
            xmin = xmax - size

        if param[1] == 'l':
            ymin = ymin_org - pad
            ymax = ymin + size
        else:
            ymax = ymax_org + pad
            ymin = ymax - size
        return xmin, xmax, ymin, ymax

    def get_single_data_from_xml(self, xml_filename, tag, idx):
        mydoc = minidom.parse(xml_filename)
        return int(mydoc.getElementsByTagName(tag)[idx].firstChild.data)

    def get_data_from_xml(self, xml_filename, idx):
        mydoc = minidom.parse(xml_filename)
        xmin = int(mydoc.getElementsByTagName('xmin')[idx].firstChild.data)
        ymin = int(mydoc.getElementsByTagName('ymin')[idx].firstChild.data)
        xmax = int(mydoc.getElementsByTagName('xmax')[idx].firstChild.data)
        ymax = int(mydoc.getElementsByTagName('ymax')[idx].firstChild.data)

        return xmin, ymin, xmax, ymax

    def get_rectangle_count(self, xml_filename):
        mydoc = minidom.parse(xml_filename)
        return len(mydoc.getElementsByTagName('xmin'))

    def get_training_values(self, img_filenames):
        x_values = []
        y_values = []
        min_gaus_nonzeros = self.screen_size ** 2

        for idx, img_filename in enumerate(img_filenames):
            xml_filename = self.get_xml_filename(img_filename)

            img = cv.imread(img_filename)
            if self.display_image_creation:
                img_display = img.copy()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            screen_pad = int(self.screen_size / 10)
            block = 251
            C = 45

            gaus = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, block, C)
            positive_mask = np.zeros(gray.shape)

            # Log True values and build positive mask
            for rect_idx in range(self.get_rectangle_count(xml_filename)):
                xmin_org, ymin_org, xmax_org, ymax_org = self.get_data_from_xml(xml_filename,
                                                                                   rect_idx)

                cv.rectangle(positive_mask, (xmin_org, ymin_org), (xmax_org, ymax_org), (255), -1)

                for corner in ["tl", "tr", "bl", "br"]:
                    xmin, xmax, ymin, ymax = self.get_roi(xmin_org, xmax_org, ymin_org, ymax_org, screen_pad,
                                                             self.screen_size, corner)
                    if xmin < 0 or xmax >= self.img_size or ymin < 0 or ymax >= self.img_size:
                        xmin, xmax, ymin, ymax = self.get_roi(xmin_org, xmax_org, ymin_org, ymax_org, 0, self.screen_size,
                                                                 corner)

                    if 0 <= xmin < xmax < 512 and 0 <= ymin < ymax < 512:
                        x_values.append(img[ymin:ymax, xmin:xmax])
                        y_values.append(True)
                        nonzero = np.count_nonzero(gaus[ymin:ymax, xmin:xmax])
                        if nonzero < min_gaus_nonzeros:
                            min_gaus_nonzeros = nonzero
                        if self.display_image_creation:
                            cv.rectangle(img_display, (xmin, ymin), (xmax, ymax), (255, 255, 0))

            # cv.imshow("mask", positive_mask)
            # cv.waitKey(0)

            # Log False values based on gaus and positive_mask
            for xmin in range(0, self.img_size, 64):
                for ymin in range(0, self.img_size, 64):
                    xmax = xmin + self.screen_size
                    ymax = ymin + self.screen_size
                    if xmax > self.img_size:
                        xmax = self.img_size
                        xmin = xmax - self.screen_size
                    if ymax > self.img_size:
                        ymax = self.img_size
                        ymin = ymax - self.screen_size
                    gaus_nonzero = np.count_nonzero(gaus[ymin:ymax, xmin:xmax])
                    positive_mask_nonzero = np.count_nonzero(positive_mask[ymin:ymax, xmin:xmax])

                    if gaus_nonzero > 100 and positive_mask_nonzero == 0:
                        x_values.append(img[ymin:ymax, xmin:xmax])
                        y_values.append(False)
                        if self.display_image_creation:
                            cv.rectangle(img_display, (xmin, ymin), (xmax, ymax), (255, 0, 255))
            if self.display_image_creation:
                cv.imshow("display_img", img_display)
                cv.imshow("zeros", positive_mask)
                cv.waitKey(0)

                        
        return x_values, y_values

    def get_xml_filename(self, img_filename):
        base_name = img_filename[-8:-4]
        # print(base_name)
        xml_filename = self.small_resources + base_name + ".xml"
        return xml_filename

    def get_positive_mask(self, img_filename):
        positive_mask = np.zeros((self.img_size, self.img_size, 1), np.uint8)
        xml_filename = self.get_xml_filename(img_filename)
        for rect_idx in range(self.get_rectangle_count(xml_filename)):
            xmin_org, ymin_org, xmax_org, ymax_org = self.get_data_from_xml(xml_filename,
                                                                            rect_idx)
            cv.rectangle(positive_mask, (xmin_org, ymin_org), (xmax_org, ymax_org), (255), -1)
        return positive_mask
