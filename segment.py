import cv2
import numpy as np
import matplotlib.pyplot as plt

row_amount = 5
shelf_height = 75
row_height = 9
space_height = 2

image = cv2.imread('test6.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_height, img_width = gray.shape

one_cm_weight = img_height / shelf_height
row_height_px = int(row_height * one_cm_weight)
space_height_px = int(space_height * one_cm_weight)

ret,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
reduced_h = cv2.reduce(th, 1, cv2.REDUCE_AVG)

largest_lines_average = int(np.average(np.sort(reduced_h, axis=0)[-row_height_px*row_amount:]))
remove_too_small_lines = reduced_h[:,0] < largest_lines_average/2
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

    # if reduced_h[i,0]:
    #     cv2.line(image, (0,i), (largest_lines_average, i), [0, 255, 0], 1)

regions = np.asarray(regions, dtype=np.int16)
get_sizes = lambda array: array[:, 1] - array[:, 0]

regions = regions[get_sizes(regions) > space_height_px]

for size in get_sizes(regions):
    print(size)

# print(regions)
#
# for coord in regions:
#     cv2.rectangle(image, (0,coord[0]), (largest_lines_average,coord[1]), (0,255,0), -1)
#
# result_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.imshow(result_image), plt.show()











# plt.subplot(121),plt.imshow(gray, cmap='gray')
# plt.subplot(122),plt.imshow(image)
# plt.show()


# cv2.line(image, (0,i), (reduced_h[i, 0], i), [0, 255, 0], 1)

# new = reduced_h[:,0]
# print(new.min(), new.max())



# blur = cv2.GaussianBlur(image,(5,5),0)

# ret,th = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# sobely = cv2.Sobel(th, cv2.CV_64F, 0, 1, ksize=5)
# sobely = cv2.Sobel(th, cv2.CV_64F, 0, 1, ksize=5)
# sobely32 = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=5)
# laplacian = cv2.Laplacian(image,cv2.CV_64F)
#
# plt.subplot(121),plt.imshow(sobely, cmap='gray')
# plt.show()

# edges = cv2.Canny(image, 80, 120)
# sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=5)
# blur = cv2.GaussianBlur(edges,(5,5),0)
# lines = cv2.HoughLinesP(image,1,np.pi/2, 500, None, 500 ,50)
#
# try:
#     for line in lines:
#         coords = line[0]
#         cv2.line(image, (coords[0],coords[1]), (coords[2],coords[3]), [255,100,100], 3)
# except:
#     pass
#
# plt.imshow(image, cmap='gray')
# plt.show()

# edges = cv2.Canny(reduced_w_graph, 100,200)
# pts = cv2.findNonZero(edges)
# rect_y = 0
# rect_height = rows
# rects = []
#
# if not len(pts):
#     rects.append(cv2.rectangle(image,(0,rect_y),(cols,rect_height), [0,255,0], 1))
#
# ref_x = 0
# ref_y = 0
#
# for i in range(len(pts)):
#     rect_height = pts[i,0,1]-ref_y
#     rects.append(cv2.rectangle(image, (0,rect_y), (cols, rect_height), [0, 255, 0], 1))
#     rect_y = pts[i,0,1]
#     ref_y = rect_y
#
#     if i == len(pts) - 1:
#         rect_height = rows - pts[i,0,1]
#         rects.append(cv2.rectangle(image, (0,rect_y), (cols, rect_height), [0, 255, 0], 1))

# print(len(rects))
