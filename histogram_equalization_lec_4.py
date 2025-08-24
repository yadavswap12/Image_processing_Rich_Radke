import numpy as np
import cv2
import matplotlib.pyplot as plt


# Read the image
img = cv2.imread('Images/hist_eqln_input.jfif')

# Check the image dimensions
print(f'Image shape is {img.shape}.')

# Convert to greyscale if necessary.
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Plot histgram of original image
hist = cv2.calcHist([img_grey], [0], None, [256], [0, 256])
# hist = cv2.calcHist([img_wash], [0], None)
plt.plot(hist)
plt.show()


# Show original image
cv2.imshow('original image', img_grey)
cv2.waitKey(0)

# Perform histogram equalization
equalized_img = cv2.equalizeHist(img_grey)

# Plot histgram of equalized image
hist = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])
# hist = cv2.calcHist([img_wash], [0], None)
plt.plot(hist)
plt.show()

# Show output image
cv2.imshow('histogram equalization', equalized_img)
cv2.waitKey(0)