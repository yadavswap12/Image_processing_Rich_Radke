import numpy as np
import cv2
import math

# Read image
# img = cv2.imread('Images/sudoku_lec_11.png')
img = cv2.imread('Images/shelves_lec_11.jfif')

# Convert image to greyscale
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Show image.
# Set window with given size to show images.
cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.imshow('Original image', img_grey)
cv2.waitKey(0)

"""
Part 1: Edege detection: Canny edge detector
"""

# Apply Canny edge detection
# edges = cv2.Canny(img_grey, 400, 500)
edges = cv2.Canny(img_grey, 100, 200)

# Show image.
# Set window with given size to show images.
cv2.namedWindow("Canny edge detector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Canny edge detector", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.imshow('Canny edge detector', edges)
cv2.waitKey(0)





"""
Part 2: Line detection: Hough transform
"""

rho_res = 1    # 1 pixel resolution
# theta_res = math.pi/180    # 1 degree resolution
theta_res = math.pi/1800    # 1 degree resolution
threshold = 105
hough_lines = cv2.HoughLines(edges, rho_res, theta_res, threshold)

# print(len(hough_lines))
print(hough_lines)




if hough_lines is not None:
    for i in range(0, len(hough_lines)):
        rho = hough_lines[i][0][0]
        theta = hough_lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(img, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
           






# Show image.
# Set window with given size to show images.
cv2.namedWindow("Line detection: Hough transform", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Line detection: Hough transform", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.imshow('Line detection: Hough transform', img)
cv2.waitKey(0)





