import numpy as np
import cv2
from skimage import transform
from skimage import filters
import seam_carving
 

"""
Part 1: Automatic saliency mapping
"""

# # Read image
# img = cv2.imread('Images/courtyard_lec_24.jpeg')

# # Convert image to greyscale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# # Display original image
# cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Original image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Original image", img_grey.shape[1]*2, img_grey.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('Original image', img_grey)

# # Create a static saliency detector (Spectral Residual)
# saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()

# # Compute the saliency map
# # The computeSaliency method expects a grayscale image, so convert if needed
# (success, saliency_map) = saliency_detector.computeSaliency(img_grey)

# # Normalize the saliency map to the 0-255 range for visualization
# saliency_map = (saliency_map * 255).astype("uint8")

# # Normalize for display
# saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)
# saliency_map = np.uint8(saliency_map)



# # Display original image
# cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Original image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Original image", img_grey.shape[1]*2, img_grey.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('Original image', img_grey)
# # cv2.waitKey(0)

# # Display saliency map
# cv2.namedWindow("Saliency Map", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Saliency Map", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Saliency Map", saliency_map.shape[1]*2, saliency_map.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('Saliency Map', saliency_map)
# # cv2.waitKey(0)

# cv2.waitKey(0)
# cv2.destroyAllWindows()



# """
# Part 2: Seam carving using skimage (currently does not work because of patent related issues)
# """


# # Read image
# img = cv2.imread('Images/courtyard_lec_24.jpeg')

# # Convert image to greyscale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# # compute the Sobel gradient magnitude representation
# # of the image -- this will serve as our "energy map"
# # input to the seam carving algorithm
# img_grey_ene_map = filters.sobel(img_grey.astype("float"))

# # Carving into square image
# # Get the shape of original image
# img_wd = img.shape[1]
# img_ht = img.shape[0]


# # Get the size of squared output image
# if img_wd > img_ht:
    # sz_wd = img_ht
    # cut_orientation = 'vertical'

# else:
    # sz_ht = img_wd
    # cut_orientation = 'horizontal'


# # perform seam carving, removing the desired number
# # of frames from the image -- `vertical` cuts will
# # change the image width while `horizontal` cuts will
# # change the image height
# img_carved = transform.seam_carve(img, img_grey_ene_map, cut_orientation, numSeams=img.shape[1]-sz_wd)


# # Display original image
# cv2.namedWindow("Energy map", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Energy map", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Energy map", img_grey_ene_map.shape[1]*2, img_grey_ene_map.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('Energy map', img_grey_ene_map)
# # cv2.waitKey(0)
    

# # Display carved image
# cv2.namedWindow("Carved image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Carved image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Carved image", img_carved.shape[1]*2, img_carved.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('Carved image', img_carved)
# # cv2.waitKey(0)

# # cv2.waitKey(0)
# # cv2.destroyAllWindows()



"""
Part 3: Seam carving using PyPI's seam_carving (See also PyPI's pyCAIR)
"""

# Read image
img = cv2.imread('Images/courtyard_lec_24.jpeg')

# Convert image to greyscale
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Carving into square image
# Get the shape of original image
img_wd = img.shape[1]
img_ht = img.shape[0]


# Get the size of squared output image
if img_wd > img_ht:
    sz_wd = img_ht
    sz_ht = sz_wd
    order='width-first'  # Choose from {width-first, height-first}
    order_rev='height-first'  # For restoring to original image size

    

else:
    sz_ht = img_wd
    sz_wd = sz_ht    
    order='height-first'  # Choose from {width-first, height-first}
    order_rev='width-first'  # For restoring to original image size


img_carved = seam_carving.resize(
    img, (sz_wd, sz_ht),
    energy_mode='backward',   # Choose from {backward, forward}
    order=order,  # Choose from {width-first, height-first}
    keep_mask=None
)


# Display original image
cv2.namedWindow("Original map", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Original map", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.resizeWindow("Original map", img.shape[1]*2, img.shape[0]*2)  # Set window to to twice the size
cv2.imshow('Original map', img)
# cv2.waitKey(0)
    

# Display carved image
cv2.namedWindow("Carved image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Carved image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.resizeWindow("Carved image", img_carved.shape[1]*2, img_carved.shape[0]*2)  # Set window to to twice the size
cv2.imshow('Carved image', img_carved)
# cv2.waitKey(0)



# Convert the square-image to original size by filling

img_filled = seam_carving.resize(
    img_carved, (img.shape[1], img.shape[0]),
    energy_mode='backward',   # Choose from {backward, forward}
    order=order_rev,  # Choose from {width-first, height-first}
    keep_mask=None
)


# Display filled image
cv2.namedWindow("Filled image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Filled image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.resizeWindow("Filled image", img_filled.shape[1]*2, img_filled.shape[0]*2)  # Set window to to twice the size
cv2.imshow('Filled image', img_filled)
# cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()



"""
Part 3: Remove an objhect using swam carving and open cv
"""

# Create an energy map using the Sobel operator
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
energy_map = np.abs(sobelx) + np.abs(sobely)

# Lower the energy of the masked object region
object_removal_mask = object_removal_mask.astype(np.float64)
energy_map[object_removal_mask == 255] = -1000 # Use a large negative number

for _ in range(n_seams_to_remove):
    # Find the minimum seam using dynamic programming
    # ... (implementation details for finding the min energy path)
    
    # Remove the seam from the image and the mask
    # ... (implementation details for removing the seam)
    
    # Update the energy map for the next iteration
    # ...




