import numpy as np
import cv2

"""
Part 1: Normal-cross-correlation and template matching
"""

# # Read image
# img = cv2.imread('Images/donuts_lec_14.jpg')
# # img_temp = cv2.imread('Images/donuts_template_lec_14.jpg')

# # If template is not available, get it as cropped section of image 
# if img is None:
    # print(f"Error: Could not read image at {image_path}")
# else:
    # # Step 2: Allow user to select ROI
    # # Display the image and allow user to select a region
    # roi = cv2.selectROI("Select ROI", img, fromCenter=False, showCrosshair=True)
    # # Press ENTER or SPACE to confirm selection, ESC to cancel

    # # Extract coordinates, width, and height
    # x, y, w, h = roi

    # if w > 0 and h > 0:  # Check if a valid ROI was selected
        # # Step 3: Crop the image
        # cropped_img = img[y:y+h, x:x+w]

        # # Step 4: Save the cropped image
        # output_path = "Images/donuts_template_lec_14.jpg"  # Desired output path
        # cv2.imwrite(output_path, cropped_img)
        # print(f"Cropped image saved to {output_path}")

        # # Optional: Display the cropped image
        # cv2.imshow("Cropped Image", cropped_img)
        # cv2.waitKey(0)
    # else:
        # print("No valid ROI selected or selection cancelled.")

    # cv2.destroyAllWindows()
    
# img_temp = cv2.imread('Images/donuts_template_lec_14.jpg')
    

# # Show original image
# # Set window with given size to show images
# cv2.namedWindow("original image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("original image 1", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("original image", img.shape[1]*2, img.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('original image', img)
# # cv2.waitKey(0)

# # Show template image
# # Set window with given size to show images
# cv2.namedWindow("template image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("template image 1", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("template image", img_temp.shape[1]*2, img_temp.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('template image', img_temp)
# # cv2.waitKey(0)

# print(img_temp.shape)

# # Get template dimensions
# # w, h = img_temp.shape[::-1]
# w, h = img_temp.shape[:-1]

# # Perform template matching with normalized cross-coefficient method
# img_out = cv2.matchTemplate(img, img_temp, cv2.TM_CCOEFF_NORMED)

# # Find the location of the best match
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img_out)

# # Get top-left corner of the best match
# top_left = max_loc

# # Calculate bottom-right corner
# bottom_right = (top_left[0] + w, top_left[1] + h)

# # Draw rectangle on the original image (color version for visualization)
# cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2) # Green rectangle, thickness 2

# # Display result

# cv2.namedWindow("Detected", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Detected", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Detected", img.shape[1]*2, img.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('Detected', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # Template matching with noisy image
# # Read image
# img = cv2.imread('Images/donuts_lec_14.jpg')
# # img_temp = cv2.imread('Images/donuts_template_lec_14.jpg')

# # Add Gaussian noise to the image.
# mean = 0
# std_dev = 40
# # gaussian_noise = np.random.normal(mean, std_dev, img.shape).astype(np.float32)
# gaussian_noise = np.random.normal(mean, std_dev, img.shape).astype(np.uint8)

# img_noise = img + gaussian_noise

# # Show noisy image.
# # Set window with given size to show images.
# cv2.namedWindow("Image with noise", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Image with noise", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Image with noise', img_noise)
# cv2.waitKey(0)


# # Perform template matching with normalized cross-coefficient method
# img_noise_out = cv2.matchTemplate(img_noise, img_temp, cv2.TM_CCOEFF_NORMED)

# # Find the location of the best match
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img_noise_out)

# # Get top-left corner of the best match
# top_left = max_loc

# # Calculate bottom-right corner
# bottom_right = (top_left[0] + w, top_left[1] + h)

# # Draw rectangle on the original image (color version for visualization)
# cv2.rectangle(img_noise, top_left, bottom_right, (0, 255, 0), 2) # Green rectangle, thickness 2

# # Display result

# cv2.namedWindow("Detected", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Detected", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Detected", img_noise.shape[1]*2, img_noise.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('Detected', img_noise)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



"""
Part 2: Normal-cross-correlation and template matching (all matching results greater than threshold)
"""

# # Read image
# img = cv2.imread('Images/building_lec_14.jpg')
# # img_temp = cv2.imread('Images/donuts_template_lec_14.jpg')


# # Convert image to greyscale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # If template is not available, get it as cropped section of image 
# if img is None:
    # print(f"Error: Could not read image at {image_path}")
# else:
    # # Step 2: Allow user to select ROI
    # # Display the image and allow user to select a region
    # roi = cv2.selectROI("Select ROI", img_grey, fromCenter=False, showCrosshair=True)
    # # Press ENTER or SPACE to confirm selection, ESC to cancel

    # # Extract coordinates, width, and height
    # x, y, w, h = roi

    # if w > 0 and h > 0:  # Check if a valid ROI was selected
        # # Step 3: Crop the image
        # cropped_img = img_grey[y:y+h, x:x+w]

        # # Step 4: Save the cropped image
        # output_path = "Images/building_template_lec_14.jpg"  # Desired output path
        # cv2.imwrite(output_path, cropped_img)
        # print(f"Cropped image saved to {output_path}")

        # # Optional: Display the cropped image
        # cv2.imshow("Cropped Image", cropped_img)
        # cv2.waitKey(0)
    # else:
        # print("No valid ROI selected or selection cancelled.")

    # cv2.destroyAllWindows()
    
# img_temp = cv2.imread('Images/building_template_lec_14.jpg')
# # Convert image to greyscale
# img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
    

# # Show original image
# # Set window with given size to show images
# cv2.namedWindow("original image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("original image 1", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("original image", img_grey.shape[1]*2, img_grey.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('original image', img_grey)
# # cv2.waitKey(0)

# # Show template image
# # Set window with given size to show images
# cv2.namedWindow("template image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("template image 1", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("template image", img_temp.shape[1]*2, img_temp.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('template image', img_temp)
# # cv2.waitKey(0)

# print(img_grey.shape)
# print(img_temp.shape)

# # Get template dimensions
# # w, h = img_temp.shape[::-1]
# # w, h = img_temp.shape[:-1]
# h, w = img_temp.shape


# # Threshold for template matching
# threshold = 0.9

# # Perform template matching with normalized cross-coefficient method
# img_out = cv2.matchTemplate(img_grey, img_temp, cv2.TM_CCOEFF_NORMED)

# loc = np.where(img_out >= threshold)

# # Store all found match locations
# match_locations = []
# for pt in zip(*loc[::-1]): # Unpack coordinates and reverse for (x, y)
    # match_locations.append(pt)
    
    # # Draw rectangles here for visualization
    # cv2.rectangle(img_grey, pt, (pt[0] + w, pt[1] + h), (255), 2)

# # Display result

# cv2.namedWindow("Detected", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Detected", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Detected", img_grey.shape[1]*2, img_grey.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('Detected', img_grey)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




"""
Part 3: Feature/corner detection
"""

# # Read image
# img = cv2.imread('Images/speed_limit_lec_14.jpeg')

# # Convert image to greyscale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Convert to float32 for cornerHarris
# img_grey = np.float32(img_grey)

# # Apply Harris Corner Detector
# # Parameters: 
# # blockSize: Size of neighborhood considered for corner detection
# # ksize: Aperture parameter of Sobel derivative used
# # k: Harris detector free parameter in the equation
# dst = cv2.cornerHarris(img_grey, blockSize=2, ksize=3, k=0.04)

# # Dilate the result to mark the corners
# dst = cv2.dilate(dst, None)

# # Threshold for optimal value, it may vary depending on the image
# # img[dst > 0.01 * dst.max()] = [0, 0, 255] # Mark corners in red
# img[dst > 0.4 * dst.max()] = [0, 0, 255] # Mark corners in red


# # Display result
# cv2.namedWindow("Harris Corners", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Harris Corners", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Harris Corners", img.shape[1]*2, img.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('Harris Corners', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# # Apply FAST-corner detection

# # # Read image
# # img = cv2.imread('Images/speed_limit_lec_14.jpeg')

# # # Convert image to greyscale
# # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # # Create FAST detector object
# # fast = cv2.FAST_create()

# # # Detect keypoints
# # kp = fast.detect(img_grey, None)

# # # Draw detected keypoints
# # img = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))

# # # Display result
# # cv2.namedWindow("FAST Corners", cv2.WINDOW_NORMAL)
# # # cv2.resizeWindow("FAST Corners", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# # cv2.resizeWindow("FAST Corners", img.shape[1]*2, img.shape[0]*2)  # Set window to to twice the size
# # cv2.imshow('FAST Corners', img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()




# # Apply SHI-TOMASI corner detection

# # Read image
# img = cv2.imread('Images/speed_limit_lec_14.jpeg')

# # Convert image to greyscale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Find corners
# # Parameters:
# # maxCorners: Maximum number of corners to return
# # qualityLevel: Parameter characterizing the minimal accepted quality of image corners
# # minDistance: Minimum possible Euclidean distance between the returned corners
# corners = cv2.goodFeaturesToTrack(img_grey, maxCorners=25, qualityLevel=0.01, minDistance=10)
# corners = np.int0(corners)

# # Draw corners
# for i in corners:
    # x, y = i.ravel()
    # cv2.circle(img, (x, y), 3, 255, -1)

# # Display result
# cv2.namedWindow("SHI-TOMASI Corners", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("SHI-TOMASI Corners", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("SHI-TOMASI Corners", img.shape[1]*2, img.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('SHI-TOMASI Corners', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





"""
Part 4: SIFT
"""

# Read image
img = cv2.imread('Images/speed_limit_lec_14.jpeg')

# Convert image to greyscale
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create a SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
kp, des = sift.detectAndCompute(img_grey, None)

# Draw keypoints on the image (optional)
# img_kp = cv2.drawKeypoints(img_grey, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_kp = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display result
cv2.namedWindow("SIFT", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("SIFT", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.resizeWindow("SIFT", img_kp.shape[1]*2, img_kp.shape[0]*2)  # Set window to to twice the size
cv2.imshow('SIFT', img_kp)
# cv2.waitKey(0)


# Read skewed image
img_skw = cv2.imread('Images/speed_limit_skewed_lec_14.jpg')

# Convert image to greyscale
img_skw_grey = cv2.cvtColor(img_skw, cv2.COLOR_BGR2GRAY)

# Create a SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
kp, des = sift.detectAndCompute(img_skw_grey, None)

# Draw keypoints on the image (optional)
# img_kp = cv2.drawKeypoints(img_grey, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_skw_kp = cv2.drawKeypoints(img_skw, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display result
cv2.namedWindow("SIFT skewed", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("SIFT", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.resizeWindow("SIFT skewed", img_skw_kp.shape[1]*2, img_skw_kp.shape[0]*2)  # Set window to to twice the size
cv2.imshow('SIFT skewed', img_skw_kp)
# cv2.waitKey(0)




# Read rotated image
img_rtd = cv2.imread('Images/speed_limit_rotated_lec_14.jpg')

# Convert image to greyscale
img_rtd_grey = cv2.cvtColor(img_rtd, cv2.COLOR_BGR2GRAY)

# Create a SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
kp, des = sift.detectAndCompute(img_rtd_grey, None)

# Draw keypoints on the image (optional)
# img_kp = cv2.drawKeypoints(img_grey, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_rtd_kp = cv2.drawKeypoints(img_rtd, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display result
cv2.namedWindow("SIFT rotated", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("SIFT", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.resizeWindow("SIFT rotated", img_rtd_kp.shape[1]*2, img_rtd_kp.shape[0]*2)  # Set window to to twice the size
cv2.imshow('SIFT rotated', img_rtd_kp)
# cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()



