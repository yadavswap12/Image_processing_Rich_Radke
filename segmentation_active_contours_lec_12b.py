import numpy as np
import cv2
import matplotlib.pyplot as plt


"""
Part 1: cv2: find contours
"""

# # Read image
# img = cv2.imread('Images/drink_segmentation_lec_12a.jfif')

# # Convert image to greyscale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# print(img_grey)

# # Show Original image.
# # Set window with given size to show images.
# cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Original image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Original image', img_grey)
# cv2.waitKey(0)

# # Apply Canny edge detection
# edges = cv2.Canny(img_grey, 100, 200)

# # Show edges image.
# # Set window with given size to show images.
# cv2.namedWindow("Edges image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Edges image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Edges image', edges)
# cv2.waitKey(0)

# contours_out, hierarchy = cv2.findContours(image=edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE) 

# # Draw all contours in green with a thickness of 2
# cv2.drawContours(img, contours_out, -1, (0, 255, 0), 2)

# # # Normalize and convert to unit8.
# # drawing_image = cv2.normalize(drawing_image, None, 0, 255, cv2.NORM_MINMAX)
# # drawing_image = np.uint8(drawing_image)

# # Show contours image.
# # Set window with given size to show images.
# cv2.namedWindow("Contours image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Contours image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Contours image', img)
# cv2.waitKey(0)



"""
Part 2: Implementing snake in sklearn
"""
from skimage.segmentation import active_contour
from skimage.filters import gaussian

# Read image
img = cv2.imread('Images/palm2_lec_12b.jfif')

# Convert image to greyscale
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Show Original image.
# # Set window with given size to show images.
# cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Original image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Original image', img)
# cv2.waitKey(0)

# # For equation based init contour (circle here) in snake
# s = np.linspace(0, 2 * np.pi, 400)
# r = 100 + 100 * np.sin(s)
# c = 220 + 100 * np.cos(s)
# init = np.array([r, c]).T


# For user drawn init contour in snake
# Global variables
points = []
drawing = False
# img = None

def draw_contour(event, x, y, flags, param):
    global points, drawing, img_grey

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points.append((x, y))
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))
            # # # Optional: Draw on the image for visual feedback
            # cv2.circle(img_grey, (x, y), 2, (0, 0, 255), -1)
            # Optional: Draw on a copy of the image for visual feedback            
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)            
            cv2.imshow("Image", img)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


# Show Original image.
# Set window with given size to show images.
cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Original image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.setMouseCallback("Original image", draw_contour)
print('Select initial conotur around object to be segmented.')
# cv2.waitKey(0)


while True:
    cv2.imshow("Original image", img_grey)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to quit and get the contour
        break

cv2.destroyAllWindows()

# init = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
# init = np.array(points, dtype=np.int32).reshape((1, 2, -1))
init = np.array(points, dtype=np.int32)
# Switch x,y as the order in mouse-event is reveresed 
init[:,0] = np.array(points, dtype=np.int32)[:, 1]
init[:,1] = np.array(points, dtype=np.int32)[:, 0]


# print(init)
# # print(np.zeros((2,2)))

# snake = active_contour(
    # img,
    # init,
    # alpha=0.015,
    # beta=10,
    # gamma=0.001,
# )

snake = active_contour(
    gaussian(img, sigma=3, preserve_range=False),
    init,
    alpha=0.015,
    beta=10,
    gamma=0.001,
)


# snake = active_contour(
    # img,
    # init,
    # alpha=0.015,
    # beta=1,
    # gamma=0.001,
    # max_iterations=1000
# )


fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img_grey.shape[1], img_grey.shape[0], 0])

plt.show()


