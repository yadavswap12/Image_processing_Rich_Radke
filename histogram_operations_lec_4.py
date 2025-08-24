import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read washed-out image.
img_wash = cv2.imread('Images/washed_out_lec_4.jfif')

# Check the shape of the image.
print(f'Image shape is {img_wash.shape}')

# Convert to greyscale if necessary.
img_wash_grey = cv2.cvtColor(img_wash, cv2.COLOR_BGR2GRAY)

# Check the shape of the greyscale image.
print(f'Greyscaled image shape is {img_wash_grey.shape}')

# Show washed-out image.
cv2.imshow('Washed-out image', img_wash_grey)
cv2.waitKey(0)

# Plot histogram of the image.
hist = cv2.calcHist([img_wash_grey], [0], None, [256], [0, 256])
# hist = cv2.calcHist([img_wash], [0], None)
plt.plot(hist)
plt.show()


# Start histogram scaling.

# Set 'High' and 'Low' values based on histogram plot above.
H = 220.0
L = 90.0

# Scale the image.
img_wash_grey_scaled = 255.0/(H-L)*(img_wash_grey-L)
# img_wash_grey_scaled = (2)*(img_wash_grey-L)

# This will truncate the decimal part and clip values outside [0, 255]
img_wash_grey_scaled = img_wash_grey_scaled.astype(np.uint8)


# Check the shape of the scaled image.
print(f'Scaled image shape is {img_wash_grey_scaled.shape}')

# Plot histogram of the scaled image.
hist_scaled = cv2.calcHist([img_wash_grey_scaled], [0], None, [256], [0, 256])
plt.plot(hist_scaled)
plt.show()

# Show scaled image.
cv2.imshow('histogram-scaled image', img_wash_grey_scaled)
cv2.waitKey(0) 