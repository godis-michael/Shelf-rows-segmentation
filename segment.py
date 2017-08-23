import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs


class SegmentWithoutQRs:
    def __init__(self, image, rows_amount=5, pack_height=9, interval=2):
        self.__image = cv2.imread(image)
        self.rows_amount = rows_amount
        self.pack_height = pack_height
        self.interval = interval

        self.__gray = cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY)
        self.__image_height, self.__image_width = self.__gray.shape

        self._shelf_height = self.rows_amount * self.pack_height + (self.rows_amount + 1) * self.interval
        self._cm_weight = self.__image_height / self._shelf_height

        self._pack_height_px = self.pack_height * self._cm_weight
        self._interval_px = interval * self._cm_weight

    def ll_amount(self, rows):

        """ Finds amount of largest lines will be used """

        return -int(self.__image_height / 2) if rows is None else -int(self._pack_height_px * self.rows_amount)
        # if rows is None:  # amount of rows is not set
        #     return -int(self.__image_height / 2)
        # else:  # amount of rows is set
        #     return -int(self._pack_height_px * self.rows_amount)

    def make_reduce(self, reduce_method=cv2.REDUCE_AVG, filtered=True, filter_coef=0.33):

        """ Create reduced graph for image """

        ret, th = cv2.threshold(self.__gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        reduced_horizontal = cv2.reduce(th, 1, reduce_method)

        if filtered:
            ll_average = int(  # average of the largest lines
                np.average(np.sort(reduced_horizontal, axis=0)[-int(5*self._pack_height_px):]))
            # set not long enough lines to 0
            too_short = reduced_horizontal[:, 0] < ll_average * filter_coef
            reduced_horizontal[too_short] = 0

        return reduced_horizontal

    def create_regions(self, array, filtered=True, filter_coef=1):

        """ Create array with found regions coordinates """

        pt1, pt2 = 0, 0
        found = False
        regions = []

        for index in range(array.shape[0]):
            if array[index, 0] and not found:  # if value is not 0 and first value was not found yet
                pt1, found = index, True
            if index == array.shape[0] - 1 and found:  # if current line is last one and first value was found
                pt2, found = index, False
                regions.append([pt1, pt2])
            elif index == array.shape[0] - 1 and not found:  # if current line is last one and first value was not found yet
                break
            elif not array[index + 1, 0] and found:  # if next value is 0 and first value was found
                pt2, found = index, False
                regions.append([pt1, pt2])

        regions = np.asarray(regions, dtype=np.int16)

        if filtered:
            sizes = lambda array: array[:, 1] - array[:, 0]
            regions = regions[sizes(regions) > self._interval_px * filter_coef]  # filter regions that are smaller than intervals

        return regions


    def __low_height(self, array, hight_coef=0.8):
        return array[1] - array[0] < self._pack_height_px * hight_coef  # check whether region is smaller than value

    def __low_interval(self, array, index, interval_coef=1.5):
        return array[index + 1, 0] - array[index, 1] < self._interval_px * interval_coef  # check whether interval is smaller than value

    def connect_regions(self, array, index=0):

        """ Connects regions that are divided """

        if index < array.shape[0] - 1:
            if self.__low_height(array[index, :]) and self.__low_height(array[index + 1, :]):
                if self.__low_interval(array, index):
                    array[index, :] = [array[index, 0], array[index + 1, 1]]
                    array = np.delete(array, index + 1, axis=0)
            index += 1
            return self.connect_regions(array, index)
        else:
            return array

    def visualise(self, array, segment=True):

        """ Visualise results """

        if array.shape[0] == self.__image_height:  # displays only reduced graph
            for index, line in enumerate(array):
                cv2.line(self.__image, (0, index), (line[0], index), (0, 255, 0), 1)
            plt.imshow(cv2.cvtColor(self.__image, cv2.COLOR_BGR2RGB))
        elif segment:  # displays divided rows separately
            if array.shape[0]:
                result_image = cv2.cvtColor(self.__image, cv2.COLOR_BGR2RGB)

                gs1 = gs.GridSpec(array.shape[0], 2)
                plt.subplot(gs1[:, 0]), plt.imshow(result_image)

                gs2 = gs.GridSpec(len(array), 2)
                for index, coord in enumerate(array):
                    plt.subplot(gs2[index, 1]), plt.imshow(result_image[coord[0]:coord[1], 0:self.__image_width])
                    plt.text(self.__image_width + 10, (coord[1] - coord[0]) / 2 + 5, index + 1)
                    plt.xticks([]), plt.yticks([])
        elif not segment:  # displays marked regions
            for coord in array:
                cv2.rectangle(self.__image, (0, coord[0]), (int(self.__image_height / 5), coord[1]), (0, 255, 0), -1)
            plt.imshow(cv2.cvtColor(self.__image, cv2.COLOR_BGR2RGB))

        plt.show()


x = SegmentWithoutQRs('drop4.jpg', rows_amount=4, interval=2)

reduced = x.make_reduce()
regions = x.create_regions(reduced)
regions = x.connect_regions(regions)
x.visualise(regions, segment=False)
