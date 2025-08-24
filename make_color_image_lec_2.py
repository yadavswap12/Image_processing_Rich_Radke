import numpy as np
import cv2

#Initialize 3D image array with zeros.
img = np.zeros((256,256,3), dtype=np.uint8)

img_arr1 = np.arange(0, 256, dtype=np.uint8).reshape(-1, 1)
img_arr2 = np.ones((1, 256), dtype=np.uint8)

img[:, :, 0] = np.matmul(img_arr1, img_arr2)

img_arr3 = np.ones((256,1), dtype=np.uint8)
img_arr4 = np.arange(0, 256, dtype=np.uint8).reshape(1, -1)

img[:, :, 2] = np.matmul(img_arr3, img_arr4)

# Show image
cv2.imshow('Image', img)

# Wait for key press
cv2.waitKey(0)

# Destroy all windows
cv2.destroyAllWindows()