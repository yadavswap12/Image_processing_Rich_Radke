import numpy as np
import cv2


"""
Part 1: For Gaussian point spread function, only few components of fft are dominant and are required for image reconstruction.
"""

# small 8X8 image with spread out Gaussian dist. of intensity
# Get a Gaussian kernel
dim_x = 8
dim_y = 8
kernel_1d_x = cv2.getGaussianKernel(dim_x, 2)
kernel_1d_y = cv2.getGaussianKernel(dim_y, 2)

img = np.outer(kernel_1d_x, kernel_1d_y.transpose())
img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
img = np.uint8(img)

# Show image
# Set window with given size to show images.
cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.imshow('Original image', img)
cv2.waitKey(0)

# fft of image
# img_fft = cv2.dft(np.float32(img_grey), flags=cv2.DFT_COMPLEX_OUTPUT)
img_fft = cv2.dft(np.float32(img), flags=cv2.DFT_REAL_OUTPUT)

print(img_fft)

# Show image
img_fft_display = cv2.normalize(img_fft, None, 0, 255, cv2.NORM_MINMAX)
img_fft_display = np.uint8(img_fft_display)

# Set window with given size to show images
cv2.namedWindow("image fft", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image fft", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.imshow('image fft', img_fft_display)
cv2.waitKey(0)



"""
Part 2: For impulse like function, many components of fft are dominant and are required for image reconstruction.
"""

# small 8X8 image with very narrow (impulse like) Gaussian dist. of intensity
# Get a Gaussian kernel
dim_x = 8
dim_y = 8
kernel_1d_x = cv2.getGaussianKernel(dim_x, 0.1)
kernel_1d_y = cv2.getGaussianKernel(dim_y, 0.1)

img = np.outer(kernel_1d_x, kernel_1d_y.transpose())
img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
img = np.uint8(img)

# Show image
# Set window with given size to show images.
cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.imshow('Original image', img)
cv2.waitKey(0)

# fft of image
# img_fft = cv2.dft(np.float32(img_grey), flags=cv2.DFT_COMPLEX_OUTPUT)
img_fft = cv2.dft(np.float32(img), flags=cv2.DFT_REAL_OUTPUT)

print(img_fft)

# Show image
img_fft_display = cv2.normalize(img_fft, None, 0, 255, cv2.NORM_MINMAX)
img_fft_display = np.uint8(img_fft_display)

# Set window with given size to show images
cv2.namedWindow("image fft", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image fft", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.imshow('image fft', img_fft_display)
cv2.waitKey(0)


"""
Part 3: For Gaussian point spread function with descrete cosine transform.
"""


# small 8X8 image with spread out Gaussian dist. of intensity
# Get a Gaussian kernel
dim_x = 8
dim_y = 8
kernel_1d_x = cv2.getGaussianKernel(dim_x, 2)
kernel_1d_y = cv2.getGaussianKernel(dim_y, 2)

img = np.outer(kernel_1d_x, kernel_1d_y.transpose())
img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
img = np.uint8(img)

# Show image
# Set window with given size to show images.
cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.imshow('Original image', img)
cv2.waitKey(0)

# fft of image
# img_fft = cv2.dft(np.float32(img_grey), flags=cv2.DFT_COMPLEX_OUTPUT)
img_dct = cv2.dct(np.float32(img), flags=None)

print(img_dct)

# Show image
img_dct_display = cv2.normalize(img_dct, None, 0, 255, cv2.NORM_MINMAX)
img_dct_display = np.uint8(img_dct)

# Set window with given size to show images
cv2.namedWindow("image dct", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image dct", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.imshow('image dct', img_dct_display)
cv2.waitKey(0)



"""
Part 4: For impulse like function with descrete cosine transform.
"""

# small 8X8 image with very narrow (impulse like) Gaussian dist. of intensity
# Get a Gaussian kernel
dim_x = 8
dim_y = 8
kernel_1d_x = cv2.getGaussianKernel(dim_x, 0.1)
kernel_1d_y = cv2.getGaussianKernel(dim_y, 0.1)

img = np.outer(kernel_1d_x, kernel_1d_y.transpose())
img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
img = np.uint8(img)

# Show image
# Set window with given size to show images.
cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.imshow('Original image', img)
cv2.waitKey(0)

# fft of image
# img_fft = cv2.dft(np.float32(img_grey), flags=cv2.DFT_COMPLEX_OUTPUT)
img_dct = cv2.dct(np.float32(img), flags=None)

print(img_dct)

# Show image
img_dct_display = cv2.normalize(img_dct, None, 0, 255, cv2.NORM_MINMAX)
img_dct_display = np.uint8(img_dct_display)

# Set window with given size to show images
cv2.namedWindow("image dct", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image dct", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.imshow('image dct', img_dct_display)
cv2.waitKey(0)
