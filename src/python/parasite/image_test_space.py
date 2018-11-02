import numpy as np
import cv2 as cv
import pandas as pd
from xml.dom import minidom


class DisplayImage:

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
