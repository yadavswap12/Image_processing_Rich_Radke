import numpy as np
import cv2
import matplotlib.pyplot as plt
import time


"""
Part 1: Limitations of simple thresholding
"""

# # Create binary image with circular foreground.
# h = 100
# w = 100
# img_bin = np.zeros((h, w))

# R = h//4

# Y, X = np.ogrid[:img_bin.shape[0], :img_bin.shape[1]]
# center = img_bin.shape[0]//2, img_bin.shape[1]//2
# dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

# mask = dist_from_center<R

# img_bin[mask] = 1

# # Show image.
# # Set window with given size to show images.
# cv2.namedWindow("Binary image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Binary image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Binary image', img_bin)
# cv2.waitKey(0)


# # Add Gaussian noise to the image.
# mean = 0
# std_dev = 0.25
# gaussian_noise = np.random.normal(mean, std_dev, img_bin.shape).astype(np.float32)

# img_bin += gaussian_noise

# # Show noisy image.
# # Set window with given size to show images.
# cv2.namedWindow("Binary image with noise", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Binary image with noise", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Binary image with noise', img_bin)
# cv2.waitKey(0)

# # Simple thresholding

# # # Plot histgram of image
# # hist = cv2.calcHist([img_bin], [0], None)
# # plt.plot(hist)
# # plt.show()


# img_bin_thresh = np.zeros(img_bin.shape)
# # mask = img_bin>0.5
# # mask = img_bin>0.1
# mask = img_bin>0.9

# img_bin_thresh[mask] = 1

# # Show thresholded image.
# # Set window with given size to show images.
# cv2.namedWindow("Thresholded image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Thresholded image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Thresholded image', img_bin_thresh)
# cv2.waitKey(0)


# # Create gradient background with circular foreground
# img_grad = np.ones((h,1))*(np.linspace(-1.0, 1.0, h).T)

# Y, X = np.ogrid[:img_grad.shape[0], :img_grad.shape[1]]
# center = img_grad.shape[0]//2, img_grad.shape[1]//2
# dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

# mask = dist_from_center<R

# img_grad[mask] = 1


# # Show gradient image.
# # Set window with given size to show images.
# cv2.namedWindow("Gradient image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Gradient image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Gradient image', img_grad)
# cv2.waitKey(0)



# # Simple thresholding

# img_grad_thresh = img_grad.copy()
# mask = img_grad>0.5
# # mask = img_grad>0.1
# # mask = img_grad>0.9

# img_grad_thresh[mask] = 1

# # Show thresholded image.
# # Set window with given size to show images.
# cv2.namedWindow("Thresholded image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Thresholded image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Thresholded image', img_grad_thresh)
# cv2.waitKey(0)



"""
Part 2: Otsu threshold
"""

# # Read image
# # img = cv2.imread('Images/sudoku_lec_11.png')
# # img = cv2.imread('Images/sunset_lec_12.jfif')
# img = cv2.imread('Images/sunset2_lec_12.jfif')

# # Convert image to greyscale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Show thresholded image.
# # Set window with given size to show images.
# cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Original image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Original image', img_grey)
# # cv2.waitKey(0)

# # Apply Otsu's binarization
# ret, otsu_thresholded = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# print(f'Otsu threshold is {ret}.')

# # Show thresholded image.
# # Set window with given size to show images.
# cv2.namedWindow("Thresholded image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Thresholded image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Thresholded image', otsu_thresholded)
# # cv2.waitKey(0)


# # Plot histgram of image
# hist = cv2.calcHist([img_grey], [0], None, [256], [0, 256])
# plt.plot(hist)
# plt.show()

# # cv2.waitKey(0)



# """
# Part 3: Adaptive thresholding
# """

# # Apply adaptive thresholding using ADAPTIVE_THRESH_MEAN_C
# img_thresh1 = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 105, 0)

# # Apply adaptive thresholding using ADAPTIVE_THRESH_GAUSSIAN_C
# img_thresh2 = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 105, 0)

# # Show thresholded image.
# # Set window with given size to show images.
# cv2.namedWindow("Adaptive thresholding with mean", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Adaptive thresholding with mean", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Adaptive thresholding with mean', img_thresh1)
# # cv2.waitKey(0)


# # Show thresholded image.
# # Set window with given size to show images.
# cv2.namedWindow("Adaptive thresholding with Gaussian weighted mean", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Adaptive thresholding with Gaussian weighted mean", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Adaptive thresholding with Gaussian weighted mean', img_thresh2)
# # cv2.waitKey(0)

# cv2.waitKey(0)



"""
Part 3: Color thresholding (segment image parts similar to given color)
"""


# Read image
img = cv2.imread('Images/fruit_stall_lec_12.jfif')



# Function for mouse-click event.
def get_pixel_color(event, x, y, flags, param):
    global col_sel
    if event == cv2.EVENT_LBUTTONDOWN:
        # Access the pixel color at the clicked coordinates
        # return img[y, x] # OpenCV stores pixels as (B, G, R)
        col_sel = img[y, x]

# # Create a window and set the mouse callback
# cv2.namedWindow('Original image')
# cv2.setMouseCallback('Original image', get_pixel_color)

# Show Original image and get pixel from mouse-click.
# Set window with given size to show images.
cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.imshow('Original image', img)
print('Image is shown.')
cv2.setMouseCallback('Original image', get_pixel_color)
print('Select the pixel in the image.')
cv2.waitKey(0)

print(col_sel)
col_int_diff = 20 


# Create mask
img_thresh = img.copy()
img_thresh_B = img_thresh[:,:,0]
img_thresh_G = img_thresh[:,:,1]
img_thresh_R = img_thresh[:,:,2]


# # Convert image to greyscale
# img_thresh = cv2.cvtColor(img_thresh, cv2.COLOR_BGR2GRAY)

print((img_thresh - col_sel).shape)

# col_sel = np.reshape(col_sel, (img_thresh.shape[0],img_thresh.shape[1],3))


# mask = abs(img_thresh - col_sel) <= col_int_diff
# mask = (img_thresh - col_sel)[:,:,0]**2 + (img_thresh - col_sel)[:,:,1]**2 + (img_thresh - col_sel)[:,:,2]*2 <= col_int_diff
# mask = np.abs(img_thresh[:,:,0] - col_sel[0])<=col_int_diff and np.abs(img_thresh[:,:,1] - col_sel[1])<=col_int_diff and np.abs(img_thresh[:,:,2] - col_sel[2])<= col_int_diff
# mask = np.abs((img_thresh - col_sel)[:,:,0])<=col_int_diff & np.abs((img_thresh - col_sel)[:,:,1])<=col_int_diff and np.abs((img_thresh - col_sel)[:,:,2])<= col_int_diff

mask_B = np.abs(img_thresh[:,:,0] - col_sel[0])<=col_int_diff
mask_G = np.abs(img_thresh[:,:,1] - col_sel[1])<=col_int_diff 
mask_R = np.abs(img_thresh[:,:,2] - col_sel[2])<= col_int_diff

# print(np.abs(img_thresh[:,:,0] - col_sel[0])<=col_int_diff)


# print(mask.shape)

# Apply mask.
img_thresh_B[~mask_B] = 0
img_thresh_B[mask_B] = 1

img_thresh_G[~mask_G] = 0
img_thresh_G[mask_G] = 1

img_thresh_R[~mask_R] = 0
img_thresh_R[mask_R] = 1

img_thresh= img_thresh_B*img_thresh_G*img_thresh_R

img_thresh = cv2.normalize(img_thresh, None, 0, 255, cv2.NORM_MINMAX)
img_thresh = np.uint8(img_thresh)



# Show Original image.
# Set window with given size to show images.
cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.imshow('Original image', img)
print('Image is shown.')
cv2.setMouseCallback('Original image', get_pixel_color)
print('Select the pixel in the image.')
# cv2.waitKey(0)

# Show Thresholded image.
# Set window with given size to show images.
cv2.namedWindow("Thresholded image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Thresholded image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.imshow('Thresholded image', img_thresh)
cv2.waitKey(0)

cv2.destroyAllWindows()







 