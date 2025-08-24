import numpy as np
import cv2

# """
# Part 1: Erosion
# """

# # Read image
# img = cv2.imread('Images/circuit_board_lec_13.jfif')

# # Convert image to greyscale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# # Create binary image by Adaptive-thresholding

# # # Apply adaptive thresholding using ADAPTIVE_THRESH_MEAN_C
# # img_thresh1 = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 105, 0)

# # Apply adaptive thresholding using ADAPTIVE_THRESH_GAUSSIAN_C
# img_bin = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 0)

# # # Show thresholded image
# # # Set window with given size to show images
# # cv2.namedWindow("Adaptive thresholding with mean", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Adaptive thresholding with mean", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# # cv2.imshow('Adaptive thresholding with mean', img_thresh1)
# # # cv2.waitKey(0)

# # Show thresholded image
# # Set window with given size to show images
# cv2.namedWindow("Adaptive thresholding with Gaussian weighted mean", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Adaptive thresholding with Gaussian weighted mean", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Adaptive thresholding with Gaussian weighted mean', img_bin)
# # cv2.waitKey(0)


# # Create structuring element
# # # Simple structuring element
# # kernel = np.ones((5,5), np.uint8)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))

# # Apply erosion
# eroded_image = cv2.erode(img_bin, kernel, iterations=1)

# # Show eroded image 
# # Set window with given size to show images
# cv2.namedWindow("Erosion with Circular element", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Erosion with Circular element", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Erosion with Circular element', eroded_image)
# # cv2.waitKey(0)

# cv2.waitKey(0)



"""
Part 2: Dilation
"""

# # Read image
# img = cv2.imread('Images/bw_text_lec_13.jfif')

# # Convert image to greyscale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Create binary image by Adaptive-thresholding

# # # Apply adaptive thresholding using ADAPTIVE_THRESH_MEAN_C
# # img_thresh1 = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 105, 0)

# # Apply adaptive thresholding using ADAPTIVE_THRESH_GAUSSIAN_C
# # img_bin = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 0)
# img_bin = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 0)    # Inverted binary image

# # Show thresholded image
# # Set window with given size to show images
# cv2.namedWindow("Adaptive thresholding with Gaussian weighted mean", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Adaptive thresholding with Gaussian weighted mean", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Adaptive thresholding with Gaussian weighted mean", img.shape[1]*2, img.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('Adaptive thresholding with Gaussian weighted mean', img_bin)
# # cv2.waitKey(0)

# # Create structuring element
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))

# # Apply erosion (to remove noise)
# eroded_image = cv2.erode(img_bin, kernel, iterations=1)

# # Show eroded image 
# # Set window with given size to show images
# cv2.namedWindow("Erosion with Circular element", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Erosion with Circular element", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Erosion with Circular element", img.shape[1]*2, img.shape[0]*2)  # Set window to twice the size
# cv2.imshow('Erosion with Circular element', eroded_image)
# # cv2.waitKey(0)

# # Create structuring element
# # # Simple structuring element
# # kernel = np.ones((3,3), np.uint8)
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2))

# # Apply erosion
# dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)

# # Show eroded image 
# # Set window with given size to show images
# cv2.namedWindow("Dilation with Circular element", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Dilation with Circular element", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Dilation with Circular element", img.shape[1]*2, img.shape[0]*2)  # Set window to twice the size
# cv2.imshow('Dilation with Circular element', dilated_image)
# # cv2.waitKey(0)

# cv2.waitKey(0)






# """
# Part 3: Opening/Closing
# """


# # Read image
# img = cv2.imread('Images/fingerprint_lec_13.jpg')

# # Convert image to greyscale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Create binary image by Adaptive-thresholding

# # # Apply adaptive thresholding using ADAPTIVE_THRESH_MEAN_C
# # img_thresh1 = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 105, 0)

# # Apply adaptive thresholding using ADAPTIVE_THRESH_GAUSSIAN_C
# # img_bin = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 0)
# img_bin = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 0)    # Inverted binary image

# # Show thresholded image
# # Set window with given size to show images
# cv2.namedWindow("Adaptive thresholding with Gaussian weighted mean", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Adaptive thresholding with Gaussian weighted mean", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Adaptive thresholding with Gaussian weighted mean", img.shape[1]*2, img.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('Adaptive thresholding with Gaussian weighted mean', img_bin)
# # cv2.waitKey(0)

# # Create structuring element
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))

# # Apply erosion (to remove noise)
# # opened_image = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)
# closed_image = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)

# # # Show eroded image 
# # # Set window with given size to show images
# # cv2.namedWindow("Opened image with Circular kernel", cv2.WINDOW_NORMAL)
# # # cv2.resizeWindow("Opened image with Circular kernel", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# # cv2.resizeWindow("Opened image with Circular kernel", img.shape[1]*2, img.shape[0]*2)  # Set window to twice the size
# # cv2.imshow('Opened image with Circular kernel', opened_image)
# # # cv2.waitKey(0)

# # Show eroded image 
# # Set window with given size to show images
# cv2.namedWindow("Closed image with Circular kernel", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Closed image with Circular kernel", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Closed image with Circular kernel", img.shape[1]*2, img.shape[0]*2)  # Set window to twice the size
# cv2.imshow('Closed image with Circular kernel', closed_image)
# # cv2.waitKey(0)

# cv2.waitKey(0)



"""
Part 4: Boundary extraction
"""


# # Read image
# img = cv2.imread('Images/bw_profile_lec_13.jpg')

# # Convert image to greyscale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Create binary image by Adaptive-thresholding

# # # # Apply adaptive thresholding using ADAPTIVE_THRESH_MEAN_C
# # # img_thresh1 = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 105, 0)

# # # Apply adaptive thresholding using ADAPTIVE_THRESH_GAUSSIAN_C
# # # img_bin = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 0)
# # img_bin = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 0)    # Inverted binary image


# # Or apply Otsu's binarization
# ret, img_bin = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



# # Show thresholded image
# # Set window with given size to show images
# cv2.namedWindow("Adaptive thresholding with Gaussian weighted mean", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Adaptive thresholding with Gaussian weighted mean", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Adaptive thresholding with Gaussian weighted mean", img.shape[1]*2, img.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('Adaptive thresholding with Gaussian weighted mean', img_bin)
# # cv2.waitKey(0)

# # Create structuring element
# # # Simple structuring element
# kernel = np.ones((3,3), np.uint8)
# # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2))
# # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))

# # Apply erosion (to get boundary removed image)
# eroded_image = cv2.erode(img_bin, kernel, iterations=1)

# # Subtract eroded image from original image to get boundary
# img_bin_boundary = img_bin - eroded_image

# # Show boundary image 
# # Set window with given size to show images
# cv2.namedWindow("Boundary image with erosion", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Closed image with Circular kernel", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Boundary image with erosion", img.shape[1]*2, img.shape[0]*2)  # Set window to twice the size
# cv2.imshow('Boundary image with erosion', img_bin_boundary)
# # cv2.waitKey(0)

# cv2.waitKey(0)







"""
Part 5: Watershed segmentation
"""

# Read image
img = cv2.imread('Images/watershed2_lec_13.png')


# # Show original image
# # Set window with given size to show images
# cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Original image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Original image", img.shape[1]*2, img.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('Original image', img)
# cv2.waitKey(0)

# Convert image to greyscale
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Show greyscale image
# Set window with given size to show images
cv2.namedWindow("Greyscale image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Greyscale image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.resizeWindow("Greyscale image", img.shape[1]*2, img.shape[0]*2)  # Set window to to twice the size
cv2.imshow('Greyscale image', img_grey)
# cv2.waitKey(0)



# Create binary image by Adaptive-thresholding

# # Apply adaptive thresholding using ADAPTIVE_THRESH_MEAN_C
# img_thresh1 = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 105, 0)

# Apply adaptive thresholding using ADAPTIVE_THRESH_GAUSSIAN_C
img_bin = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 0)
# img_bin = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 0)    # Inverted binary image


# Show binary image
# Set window with given size to show images
cv2.namedWindow("Binary image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Binary image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.resizeWindow("Binary image", img.shape[1]*2, img.shape[0]*2)  # Set window to to twice the size
cv2.imshow('Binary image', img_bin)
# cv2.waitKey(0)




dt = cv2.distanceTransform(img_bin, 2, 3)
# dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.int32)

# _, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)
# lbl, ncc = label(dt)
# lbl = lbl * (255 / (ncc + 1))
# Completing the markers now. 
# lbl[border == 255] = 255

# lbl = lbl.astype(numpy.int32)
# cv2.watershed(a, lbl)


# Apply watershed
img_wtrshd = cv2.watershed(img, dt)

# Normalize and convert to unit8.
img_wtrshd = cv2.normalize(img_wtrshd, None, 0, 255, cv2.NORM_MINMAX)
img_wtrshd = np.uint8(img_wtrshd)

# Show watershed image 
# Set window with given size to show images
cv2.namedWindow("Watershed segmentation", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Watershed segmentation", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.resizeWindow("Watershed segmentation", img.shape[1]*2, img.shape[0]*2)  # Set window to twice the size
cv2.imshow('Watershed segmentation', img_wtrshd)
# cv2.waitKey(0)

cv2.waitKey(0)








