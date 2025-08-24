import numpy as np
import cv2

img_arr1 = np.arange(0, 256, dtype=np.uint8).reshape(-1, 1)
img_arr2 = np.ones((1,256), dtype=np.uint8)
img = np.matmul(img_arr1, img_arr2)

# Show image
cv2.imshow('Image', img)

# Wait for key press
cv2.waitKey(0)

# Destroy all windows
cv2.destroyAllWindows()