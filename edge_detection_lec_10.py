import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('Images/edge_detection_lec_10.jfif')

# Convert image to greyscale
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Show image.
# Set window with given size to show images.
cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.imshow('Original image', img_grey)
cv2.waitKey(0)


"""
Part 1: Crude edge detector 
"""

# 2D spatial filter: edge detector
F_x = np.array([[-1, 1]])    # Vertical edge-detector
F_y = np.array([[-1], [1]])    # horizontal edge-detector

# Apply filter.
img_filt_x = cv2.filter2D(img_grey, ddepth=-1, kernel=F_x, dst=None, anchor=None, delta=None, borderType=None)
img_filt_y = cv2.filter2D(img_grey, ddepth=-1, kernel=F_y, dst=None, anchor=None, delta=None, borderType=None)

# Show image.
# Set window with given size to show images.
cv2.namedWindow("Vertical-edge filter", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vertical-edge filter", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.imshow('Vertical-edge filter', img_filt_x)
cv2.waitKey(0)

# Show image.
# Set window with given size to show images.
cv2.namedWindow("horizontal-edge filter", cv2.WINDOW_NORMAL)
cv2.resizeWindow("horizontal-edge filter", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.imshow('horizontal-edge filter', img_filt_y)
cv2.waitKey(0)

# Create a mask for pixels above the threshold in grayscale
threshold_value = 100
mask_x = img_filt_x > threshold_value
mask_y = img_filt_y > threshold_value
img_filt_x_masked = np.zeros(img_filt_x.shape)
img_filt_x_masked[mask_x] = img_filt_x[mask_x]
img_filt_y_masked = np.zeros(img_filt_y.shape)
img_filt_y_masked[mask_y] = img_filt_y[mask_y]

# Show image.
# Set window with given size to show images.
cv2.namedWindow("Vertical-edge filter with threshold", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vertical-edge filter with threshold", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.imshow('Vertical-edge filter with threshold', img_filt_x_masked)
cv2.waitKey(0)

# Show image.
# Set window with given size to show images.
cv2.namedWindow("horizontal-edge filter with threshold", cv2.WINDOW_NORMAL)
cv2.resizeWindow("horizontal-edge filter with threshold", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.imshow('horizontal-edge filter with threshold', img_filt_y_masked)
cv2.waitKey(0)



# """
# Part 2:  Edge detector: Sobel filter 
# """

# # 2D spatial filter: edge detector
# F_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])    # Vertical edge-detector
# F_y = F_x.T    # horizontal edge-detector

# # Apply filter.
# img_filt_x = cv2.filter2D(img_grey, ddepth=-1, kernel=F_x, dst=None, anchor=None, delta=None, borderType=None)
# img_filt_y = cv2.filter2D(img_grey, ddepth=-1, kernel=F_y, dst=None, anchor=None, delta=None, borderType=None)

# # Show image.
# # Set window with given size to show images.
# cv2.namedWindow("Vertical-edge filter", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Vertical-edge filter", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Vertical-edge filter', img_filt_x)
# cv2.waitKey(0)

# # Show image.
# # Set window with given size to show images.
# cv2.namedWindow("horizontal-edge filter", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("horizontal-edge filter", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('horizontal-edge filter', img_filt_y)
# cv2.waitKey(0)

# # Create a mask for pixels above the threshold in grayscale
# threshold_value = 100
# mask_x = img_filt_x > threshold_value
# mask_y = img_filt_y > threshold_value
# img_filt_x_masked = np.zeros(img_filt_x.shape)
# img_filt_x_masked[mask_x] = img_filt_x[mask_x]
# img_filt_y_masked = np.zeros(img_filt_y.shape)
# img_filt_y_masked[mask_y] = img_filt_y[mask_y]

# # Show image.
# # Set window with given size to show images.
# cv2.namedWindow("Vertical-edge filter with threshold", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Vertical-edge filter with threshold", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Vertical-edge filter with threshold', img_filt_x_masked)
# cv2.waitKey(0)

# # Show image.
# # Set window with given size to show images.
# cv2.namedWindow("horizontal-edge filter with threshold", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("horizontal-edge filter with threshold", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('horizontal-edge filter with threshold', img_filt_y_masked)
# cv2.waitKey(0)



# """
# Part 3: Magnitude and angle of edge-activity 
# """

# img_filt_mag = (img_filt_x**2 + img_filt_y**2)**(0.5)
# eps = 1e-20
# img_filt_ang = np.arctan(img_filt_y/(img_filt_x+eps))

# img_filt_mag_disp = cv2.normalize(img_filt_mag, None, 0, 255, cv2.NORM_MINMAX)
# img_filt_mag_disp = np.uint8(img_filt_mag_disp)

# img_filt_ang_disp = cv2.normalize(img_filt_ang, None, 0, 255, cv2.NORM_MINMAX)
# img_filt_ang_disp = np.uint8(img_filt_ang_disp)

# # Show image.
# # # Set window with given size to show images.
# # cv2.namedWindow("image filter magnitude", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("image filter magnitude", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# # cv2.imshow('image filter magnitude', img_filt_mag_disp)
# # cv2.waitKey(0)

# plt.imshow(img_filt_mag_disp, cmap='viridis')  # 'viridis' is a perceptually uniform colormap
# plt.colorbar()  # Add a color bar to indicate intensity mapping
# plt.title('image filter magnitude')
# plt.show()

# # Show image.
# # # Set window with given size to show images.
# # cv2.namedWindow("image filter direction", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("image filter direction", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# # cv2.imshow('image filter direction', img_filt_ang_disp)
# # cv2.waitKey(0)

# plt.imshow(img_filt_mag_disp, cmap='viridis')  # 'viridis' is a perceptually uniform colormap
# plt.colorbar()  # Add a color bar to indicate intensity mapping
# plt.title('image filter direction')
# plt.show()



# """
# Part 4: Laplacian of Gaussian filter, aka Maar-Hildreth filter, aka Mexican-hat filter
# """

# # Reshape to square image
# img_grey = cv2.resize(img_grey, (512,512))


# # 2D spatial filter:  Laplacian of Gaussian filter, aka Maar-Hildreth filter, aka Mexican-hat filter
# # Get a Gaussian kernel
# dim_x = img_grey.shape[0]
# dim_y = img_grey.shape[1]
# sig = 8
# kernel_1d_x = cv2.getGaussianKernel(dim_x, sig)
# kernel_1d_y = cv2.getGaussianKernel(dim_y, sig)

# # Get prefactor
# eps = 1e-20
# prefactor_x = np.log(kernel_1d_x+eps)*(-2.0/sig**2)
# prefactor_y = np.log(kernel_1d_y+eps)*(-2.0/sig**2)
# prefactor = prefactor_x + prefactor_y - (2.0/sig**2)
 
# # To create a 2D Gaussian kernel (e.g., for direct convolution)
# # This involves multiplying the 1D kernel with its transpose
# F = np.outer(kernel_1d_x, kernel_1d_y.transpose())

# F = prefactor*F

# F_disp = cv2.normalize(F, None, 0, 255, cv2.NORM_MINMAX)
# F_disp = np.uint8(F_disp)

# # Show image of filter.
# # Set window with given size to show images.
# cv2.namedWindow("Laplacian of Gaussian filter", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Laplacian of Gaussian filter", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Laplacian of Gaussian filter', F_disp)
# cv2.waitKey(0)

# # Apply filter.
# img_filt = cv2.filter2D(img_grey, ddepth=-1, kernel=F, dst=None, anchor=None, delta=None, borderType=None)

# img_filt_disp = cv2.normalize(img_filt, None, 0, 255, cv2.NORM_MINMAX)
# img_filt_disp = np.uint8(img_filt_disp)

# # Show image.
# # Set window with given size to show images.
# cv2.namedWindow("Laplacian of Gaussian filter", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Laplacian of Gaussian filter", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Laplacian of Gaussian filter', img_filt_disp)
# cv2.waitKey(0)





# """
# Part 5: Difference of Gaussian filter (approximation to Laplacian of Gaussian filter)
# """

# # Reshape to square image
# img_grey = cv2.resize(img_grey, (512,512))


# # 2D spatial filter:  Laplacian of Gaussian filter, aka Maar-Hildreth filter, aka Mexican-hat filter
# # Get a Gaussian kernel
# dim_x = img_grey.shape[0]
# dim_y = img_grey.shape[1]

# sig_1 = 16
# kernel_1_1d_x = cv2.getGaussianKernel(dim_x, sig_1)
# kernel_1_1d_y = cv2.getGaussianKernel(dim_y, sig_1)

# sig_2 = 2
# kernel_2_1d_x = cv2.getGaussianKernel(dim_x, sig_2)
# kernel_2_1d_y = cv2.getGaussianKernel(dim_y, sig_2)
 
# # To create a 2D Gaussian kernel (e.g., for direct convolution)
# # This involves multiplying the 1D kernel with its transpose
# F1 = np.outer(kernel_1_1d_x, kernel_1_1d_y.transpose())
# F2 = np.outer(kernel_2_1d_x, kernel_2_1d_y.transpose())

# F = F2-F1

# F_disp = cv2.normalize(F, None, 0, 255, cv2.NORM_MINMAX)
# F_disp = np.uint8(F_disp)

# # Show image of filter.
# # Set window with given size to show images.
# cv2.namedWindow("Difference of Gaussian filter", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Difference of Gaussian filter", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Difference of Gaussian filter', F_disp)
# cv2.waitKey(0)

# # Apply filter.
# img_filt = cv2.filter2D(img_grey, ddepth=-1, kernel=F, dst=None, anchor=None, delta=None, borderType=None)

# img_filt_disp = cv2.normalize(img_filt, None, 0, 255, cv2.NORM_MINMAX)
# img_filt_disp = np.uint8(img_filt_disp)

# # Show image.
# # Set window with given size to show images.
# cv2.namedWindow("Difference of Gaussian filter", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Difference of Gaussian filter", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Difference of Gaussian filter', img_filt_disp)
# cv2.waitKey(0)








"""
Part 5: Canny edge detector
"""

# Apply Canny edge detection
edges = cv2.Canny(img_grey, 400, 500)

# Show image.
# Set window with given size to show images.
cv2.namedWindow("Canny edge detector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Canny edge detector", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.imshow('Canny edge detector', edges)
cv2.waitKey(0)









