import numpy as np
import cv2

# Read image
img = cv2.imread('Images/noisy_lec_6.jfif')

# Show input image.
cv2.imshow('original image', img)
cv2.waitKey(0)

# 2D spatial filter: smoothing (low pass filter)
F = (1.0/9)*np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
# F = (1.0/36)*np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])


# Apply filter.
img_filt = cv2.filter2D(img, ddepth=-1, kernel=F, dst=None, anchor=None, delta=None, borderType=None)

# Show image.
cv2.imshow('smoothed image', img_filt)
cv2.waitKey(0)

# 2D spatial filter: lowpass
F = (1.0/16)*np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

# Apply filter.
img_filt = cv2.filter2D(img, ddepth=-1, kernel=F, dst=None, anchor=None, delta=None, borderType=None)

# Show image.
cv2.imshow('low-pass filtered image', img_filt)
cv2.waitKey(0)

# 2D spatial filter: edge detection
F = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

# Read image
img = cv2.imread('Images/sharpenning_lec_6.jfif')

# Show input image.
cv2.imshow('original image', img)
cv2.waitKey(0)

# Apply filter.
img_filt = cv2.filter2D(img, ddepth=-1, kernel=F, dst=None, anchor=None, delta=None, borderType=None)

# # This will truncate the decimal part and clip values outside [0, 255]
# img_filt = img_filt.astype(np.uint8)

# Show image.
cv2.imshow('edge detected image', img_filt)
cv2.waitKey(0)

# Show image with pixels above given threshold.
threshold_value = 200 
# Create a mask for pixels above the threshold in grayscale
mask = img_filt > threshold_value

# Create an empty image with the same dimensions as the original
img_filt_thresh = np.zeros_like(img_filt)

# Copy pixels from the original image to the result_image where the mask is True
img_filt_thresh[mask] = img_filt[mask]

cv2.imshow('edge detected image (thresholded)', img_filt_thresh)
cv2.waitKey(0)

# 2D spatial filter: sharpenning
F = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) + np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) # original pixel + filter for edge detection. 

# Make smoothed image from original sharp image.
F_smooth = (1.0/9)*np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])    # 2D spatial filter: smoothing
img_smooth = cv2.filter2D(img, ddepth=-1, kernel=F_smooth, dst=None, anchor=None, delta=None, borderType=None)    # Apply filter
cv2.imshow('smoothened image', img_smooth)    # Show smoothened image.
cv2.waitKey(0)

# Apply filter.
img_filt = cv2.filter2D(img_smooth, ddepth=-1, kernel=F, dst=None, anchor=None, delta=None, borderType=None)

# Show image.
cv2.imshow('sharpenned image', img_filt)
cv2.waitKey(0)

# 2D spatial filter: Un-sharp
# Un-sharp filter = original pixel filter + (para_lambda)*high-pass-filter
# high-pass-filter = original pixel filter - low-pass-filter
para_lambda = 0.8
F_pix = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
F_low_pass = (1.0/9)*np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
F_high_pass = F_pix - F_low_pass
F = F_pix + para_lambda*F_high_pass

# Apply filter.
img_filt = cv2.filter2D(img_smooth, ddepth=-1, kernel=F, dst=None, anchor=None, delta=None, borderType=None)

# Show image.
cv2.imshow('un-sharp filter image', img_filt)
cv2.waitKey(0)

# 2D spatial filter: Sobel-horizontal edge detector
F = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Apply filter.
img_filt = cv2.filter2D(img, ddepth=-1, kernel=F, dst=None, anchor=None, delta=None, borderType=None)

# Show image.
cv2.imshow('Sobel-horizontal-edge filter', img_filt)
cv2.waitKey(0)

# 2D spatial filter: Sobel-vertical edge detector
F = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

# Apply filter.
img_filt = cv2.filter2D(img, ddepth=-1, kernel=F, dst=None, anchor=None, delta=None, borderType=None)

# Show image.
cv2.imshow('Sobel-vertical-edge filter', img_filt)
cv2.waitKey(0)


# 2D spatial filter: Median filter (for salt-and-pepper noise)

# Read image.
img = cv2.imread('Images/salt_and_pepper_lec_6.png')

# Show image.
cv2.imshow('Original image', img)
cv2.waitKey(0)

# Apply median blur with a kernel size of 5
img_filt = cv2.medianBlur(img, 5)

# Show image.
cv2.imshow('Median filter', img_filt)
cv2.waitKey(0)

  










