import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs


row_amount = 7
row_height = 9
space_height = 4
shelf_height = (row_amount * row_height) + ((row_amount+1) * space_height)

image = cv2.imread('test6.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_height, img_width = gray.shape

one_cm_weight = img_height / shelf_height
row_height_px = int(row_height * one_cm_weight)
space_height_px = int(space_height * one_cm_weight)

ret,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
reduced_h = cv2.reduce(th, 1, cv2.REDUCE_AVG)

largest_lines_average = int(np.average(np.sort(reduced_h, axis=0)[-row_height_px*row_amount:]))
remove_too_small_lines = reduced_h[:,0] < largest_lines_average/3
reduced_h[remove_too_small_lines] = 0

pt1,pt2 = 0,0
found = False
regions = []


for i in range(img_height):
    if reduced_h[i,0] and not found:
        pt1 = i
        found = True
    if i == img_height-1 and found:
        pt2 = i
        regions.append([pt1, pt2])
        found = False
    elif i == img_height-1 and not found:
        break
    elif not reduced_h[i+1,0] and found:
        pt2 = i
        regions.append([pt1, pt2])
        found = False

regions = np.asarray(regions, dtype=np.int16)
get_sizes = lambda array: array[:, 1] - array[:, 0]

regions = regions[get_sizes(regions) > space_height_px]

def not_high_enough(array):
    return array[1] - array[0] < row_height_px * 0.9

def space_is_low(array, index):
    return array[index + 1, 0] - array[index, 1] < space_height_px * 1.5

def connect_regions(regions, index=0):
    if index < regions.shape[0] - 1:
        if not_high_enough(regions[index, :]) and not_high_enough(regions[index + 1, :]):
            if space_is_low(regions, index):
                regions[index, :] = [regions[index, 0], regions[index + 1, 1]]
                regions = np.delete(regions, index + 1, axis=0)
        index += 1
        return connect_regions(regions, index)
    else:
        return regions

graph = connect_regions(regions)
result_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gs1 = gs.GridSpec(len(graph),2)
plt.subplot(gs1[:,0]), plt.imshow(result_image)

if len(graph):
    gs2 = gs.GridSpec(len(graph), 2)
    for index,coord in enumerate(graph):
        plt.subplot(gs2[index, 1]), plt.imshow(result_image[coord[0]:coord[1], 0:img_width])
        plt.text(img_width + 10, (coord[1]-coord[0])/2 + 5, index + 1)
        plt.xticks([]), plt.yticks([])

plt.show()

