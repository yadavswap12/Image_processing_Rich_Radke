import numpy as np
import cv2

# Read image
img = cv2.imread('Images/visible_light_spectrum.png')

# Show original image
cv2.imshow('Original image', img)
cv2.waitKey(0)

# Define Affine transformation matrix
M = np.array([[0.4, 0.2, 4], [0.1, 0.6, 2]])

# Apply affine transformation
img_aff_tx = cv2.warpAffine(img, M, dsize=(img.shape[0], img.shape[1]))

# Show transformed image
cv2.imshow('Warped image', img_aff_tx)
cv2.waitKey(0)

# # Define Perspective transformation matrix
# # M = np.array([[1.0, 0.0, 0.001], [0.0, 0.1, 0], [-200, -300, 1]])
# M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

# Define source points (e.g., corners of a document in the image)
# These should be in the order: top-left, top-right, bottom-right, bottom-left
x1=0
y1=0 
x2=0 
y2=img.shape[1]
x3=img.shape[0]
y3=img.shape[1]
x4=img.shape[0]
y4=0

src_pts = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

# Define destination points (e.g., a perfect rectangle)
# These should correspond to the order of src_pts
width, height = img.shape[0], img.shape[1] # Desired output dimensions
dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

# Get the perspective transformation matrix
M = cv2.getPerspectiveTransform(src_pts, dst_pts)


# Apply perspective transformation
img_prsp_tx = cv2.warpPerspective(img, M, dsize=(img.shape[0], img.shape[1]))

# Show transformed image
cv2.imshow('tranformed image', img_prsp_tx)
cv2.waitKey(0)