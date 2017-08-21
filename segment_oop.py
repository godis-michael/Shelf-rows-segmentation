import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs


class SegmentWithoutQRs:
    def __init__(self, image, rows_amount=None, pack_height=9, interval=2):
        self.image = cv2.imread(image)
        self.rows_amount = rows_amount
        self.pack_height = pack_height
        self.interval = interval

        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.image_height, self.image_width = self.gray.shape

        self.shelf_height = self.rows_amount * self.pack_height + (self.rows_amount + 1) * self.interval
        self.cm_weight =  self.image_height / self.shelf_height

        self.pack_height_px = self.pack_height * self.cm_weight
        self.interval_px = interval * self.cm_weight

    def make_reduce(self, reduce_method=cv2.REDUCE_AVG, filtered=True, filter_value=0.33):
        ret, th = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        reduced_horizontal = cv2.reduce(th, 1, reduce_method)

        if filtered:
            # average of the largest lines
            ll_average = int(np.average(np.sort(reduced_horizontal, axis=0)[-self.pack_height_px*self.rows_amount:]))

x = SegmentWithoutQRs('test6.jpg')
