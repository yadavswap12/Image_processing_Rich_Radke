import numpy as np
import cv2

"""
Part 1: Suparpixels with SLIC.
"""

# # Read image
# img = cv2.imread('Images/drink_segmentation_lec_12a.jfif')

# # Show Original image.
# # Set window with given size to show images.
# cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Original image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Original image', img)
# cv2.waitKey(0)

# # Preprocessing (recommended for color images)
# img_lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB)

# # Show Original image.
# # Set window with given size to show images.
# cv2.namedWindow("Processed image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Processed image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Processed image', img_lab)
# cv2.waitKey(0)

# # Create SuperpixelSLIC object
# # algorithm: SLIC, SLICO, or MSLIC (SLICO is an optimization of SLIC, MSLIC is manifold SLIC)
# # region_size: Approximate size of each superpixel
# # ruler: Enforcement of superpixel smoothness (0 to 100, where 10 is recommended)
# slic = cv2.ximgproc.createSuperpixelSLIC(img_lab, algorithm=cv2.ximgproc.SLICO, region_size=10, ruler=10)

# # Iterate to compute the superpixels
# slic.iterate()

# # Get the segmentation labels
# labels = slic.getLabels()

# # print(f'labels are {labels}') 

# # You can then use these labels to visualize the superpixels, 
# # extract features, or for further processing. 
# # For example, to get the superpixel boundaries:
# mask = slic.getLabelContourMask()

# # Overlay superpixel boundaries on the original image
# # img_with_boundaries = cv2.bitwise_and(img_lab, img_lab, mask=mask)
# img_with_boundaries = cv2.bitwise_and(img_lab, img_lab, mask=~mask)

# # Show image with supaerpixels.
# # Set window with given size to show images.
# cv2.namedWindow("Superpixels image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Superpixels image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Superpixels image', img_with_boundaries)
# cv2.waitKey(0)

# cv2.destroyAllWindows()


"""
Part 2: Suparpixels with SEEDS.
"""


# # Read image
# img = cv2.imread('Images/drink_segmentation_lec_12a.jfif')

# # Show Original image.
# # Set window with given size to show images.
# cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Original image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Original image', img)
# cv2.waitKey(0)


# # Create SuperpixelSEEDS object
# # img_width, img_height, img_channels: Dimensions of the image
# # num_superpixels: Desired number of superpixels
# # num_levels: Number of block levels (higher value = more accuracy, but more time)
# # use_prior: Enable 3x3 shape smoothing term (larger value = smoother shapes, 0 to 5)
# # num_histogram_bins: Number of histogram bins
# # double_step: Iterate each block level twice for higher accuracy
# seeds = cv2.ximgproc.createSuperpixelSEEDS(img.shape[1], img.shape[0], img.shape[2], 
                                          # num_superpixels=500, num_levels=4, 
                                          # prior=1, double_step=True)
                                          
# # # Set the histogram bins *after* initialization
# # seeds.setHistogramBins(5)  # Example value, adjust as needed

# # Iterate to compute the superpixels
# seeds.iterate(img)

# # Get the segmentation labels
# labels = seeds.getLabels()

# # Get the superpixel boundaries mask
# mask = seeds.getLabelContourMask()

# # Overlay superpixel boundaries on the original image
# # img_with_boundaries = cv2.bitwise_and(img_lab, img_lab, mask=mask)
# img_with_boundaries = cv2.bitwise_and(img, img, mask=~mask)

# # Show image with supaerpixels.
# # Set window with given size to show images.
# cv2.namedWindow("Superpixels image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Superpixels image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Superpixels image', img_with_boundaries)
# cv2.waitKey(0)

# cv2.destroyAllWindows()





"""
Part 3: Graph-cut.
"""

# Read image
img = cv2.imread('Images/koala_segmentation_lec_12a.jfif')

# Show Original image.
# Set window with given size to show images.
cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.imshow('Original image', img)
cv2.waitKey(0)

mask = np.zeros(img.shape[:2], np.uint8)
# mask = np.zeros(img.shape, np.uint8)

# Get foreground bb from user (the bb should fully cover the foreground).
r = cv2.selectROI("Select foreground bounding-box which fully covers the foreground", img, fromCenter=False, showCrosshair=True)

# Initialize background and foreground models (always like this for any input)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)


# graph_cut_out = cv2.grabCut(img, mask=mask, rect=r, bgdModel=bgd_model, fgdModel=fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
cv2.grabCut(img, mask=mask, rect=r, bgdModel=bgd_model, fgdModel=fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

# Create a new mask where probable background and definite background are set to 0 (background)
# and probable foreground and definite foreground are set to 1 (foreground).
output_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')

# print(np.where(graph_cut_out[0]==1))
# print(output_mask)
# print(graph_cut_out[0].shape)
# print(graph_cut_out[1].shape)
# print(graph_cut_out[2].shape)


# Overlay superpixel boundaries on the original image
img_with_segmentation = cv2.bitwise_and(img, img, mask=output_mask)
# img_with_segmentation = cv2.bitwise_and(img, img, mask=~mask)

# img_with_segmentation = cv2.normalize(img_with_segmentation, None, 0, 255, cv2.NORM_MINMAX)
# img_with_segmentation = np.uint8(img_with_segmentation)


# print(np.where(img_with_segmentation==1))

# Show image with supaerpixels.
# Set window with given size to show images.
cv2.namedWindow("Graph-cut image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Graph-cut image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.imshow('Graph-cut image', img_with_segmentation)
cv2.waitKey(0)

cv2.destroyAllWindows()
