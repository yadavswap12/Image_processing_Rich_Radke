import numpy as np
import cv2

"""
Part 3: PDE based inpainting
"""

# Read image
img = cv2.imread('Images/gondola_lec_22.png')

# # Convert image to greyscale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Make sure the source and target images are of same size and are np.int32 (for compatible addition)
img_width = 512
img_height = 512

# img = cv2.resize(img, (img_width, img_height)).astype(np.int32)
img = cv2.resize(img, (img_width, img_height))

img_copy = img.copy().astype(np.uint8)

# # Convert image to greyscale
# img_src_grey = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

# Create a mask for object in source image
mask = np.zeros(img.shape[:2], dtype="uint8")

# For user drawn ROI contour
points = []
drawing = False

def draw_contour(event, x, y, flags, param):
    global points, drawing, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points.append((x, y))
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))
            # # Optional: Draw on a copy of the image for visual feedback
            cv2.circle(img_copy, (x, y), 2, (0, 0, 255), -1)
            cv2.imshow("Image", img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


# Show Original image.
# Set window with given size to show images.
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.setMouseCallback("image", draw_contour)
print('Select initial conotur around object to be segmented.')
# cv2.waitKey(0)


while True:
    cv2.imshow("image", img.astype(np.uint8))
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to quit and get the contour
        break


cntr_pts = np.array(points, dtype=np.int32)
# Switch x,y as the order in mouse-event is reveresed 
cntr_pts = cntr_pts.reshape((-1, 1, 2))


# Draw the polygon on the mask, filling it with white (255)
# cv2.fillPoly(mask, [polygon_points], 255)
cv2.fillPoly(mask, [cntr_pts], 255)
# cv2.fillPoly(mask, [cntr_pts], 1)

# 3-channel mask
# mask = mask.astype(np.float32)

# # Reshape mask for color image
# mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)


# Inpaint using TELEA method
inpainted_telea = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

# Inpaint using NS method
inpainted_ns = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

# Show inpainting image.

# Set window with given size to show images.
cv2.namedWindow("TELEA inpainted image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("TELEA inpainted image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.resizeWindow("TELEA inpainted image", inpainted_telea.shape[1]*1, inpainted_telea.shape[0]*1)  # Set window to to twice the size
cv2.imshow('TELEA inpainted image', inpainted_telea.astype(np.uint8))

# Set window with given size to show images.
cv2.namedWindow("NS inpainted image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("NS inpainted image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.resizeWindow("NS inpainted image", inpainted_ns.shape[1]*1, inpainted_ns.shape[0]*1)  # Set window to to twice the size
cv2.imshow('NS inpainted image', inpainted_ns.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()


