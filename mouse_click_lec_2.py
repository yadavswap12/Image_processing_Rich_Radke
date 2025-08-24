import numpy as np
import cv2


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinates: ({x}, {y})")
        print(f"Colors: Blue={img[y][x][0]}, Green={img[y][x][1]}, Red={img[y][x][2]}")
        print(f"Colors: {img[y][x]}")
        # cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        # cv2.imshow("Image", img)


# img_file = 'Images/lecture_2.png'
img_file = 'Images/visible_light_spectrum.png'

img = cv2.imread(img_file) 

# print(type(img))
print(img.shape)


# Display image in window
cv2.imshow('Image', img)


cv2.setMouseCallback("Image", click_event)


# Wait for a key press
cv2.waitKey(0)


# Destroy all windows
cv2.destroyAllWindows()