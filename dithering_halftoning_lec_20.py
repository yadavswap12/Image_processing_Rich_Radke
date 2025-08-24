import numpy as np
import cv2





"""
Part 1: Simple dithering by thresholding
"""
# # Read image
# img = cv2.imread('Images/parade_lec_20.jpg')

# # Convert image to greyscale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# # Simple thresholding
# img_bin_thresh = np.zeros(img_grey.shape)
# mask = img_grey>128

# img_bin_thresh[mask] = 1

# # Display original image
# cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Original image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Original image", img_grey.shape[1]*2, img_grey.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('Original image', img_grey)

# # Show thresholded image.
# # Set window with given size to show images.
# cv2.namedWindow("Thresholded image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Thresholded image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Thresholded image', img_bin_thresh)
# cv2.waitKey(0)


"""
Part 2: Noise + dithering by thresholding
"""

# # Read image
# img = cv2.imread('Images/parade_lec_20.jpg')

# # Convert image to greyscale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Add Gaussian noise to the image.
# mean = 0
# std_dev = 8
# gaussian_noise = np.random.normal(mean, std_dev, img_grey.shape).astype(np.float32)
# # gaussian_noise = np.random.normal(mean, std_dev, img_grey.shape).astype(np.uint8)

# img_noise = img_grey + gaussian_noise

# # Simple thresholding
# img_noise_bin_thresh = np.zeros(img_noise.shape)
# mask = img_noise>128

# img_noise_bin_thresh[mask] = 1

# # Show noisy image.
# img_noise = cv2.normalize(img_noise, None, 0, 255, cv2.NORM_MINMAX)
# img_noise = np.uint8(img_noise)

# # Set window with given size to show images.
# cv2.namedWindow("Image with noise", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Image with noise", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Image with noise', img_noise)

# # Show thresholded image.
# # Set window with given size to show images.
# cv2.namedWindow("Thresholded image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Thresholded image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Thresholded image', img_noise_bin_thresh)
# cv2.waitKey(0)



"""
Part 3: Halftoning by ordered dithering
"""

# # Read image
# img = cv2.imread('Images/parade_lec_20.jpg')

# # Convert image to greyscale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# # Reshape image to the size which is multiple of 5
# new_width = 1000
# new_height = 1000
# img_grey = cv2.resize(img_grey, (new_width, new_height))

# # Display resized image
# cv2.namedWindow("Resized image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Resized image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Resized image", img_grey.shape[1]*2, img_grey.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('Resized image', img_grey)


# # Convert image to float
# img_grey = img_grey.astype(np.float32)

# # Create halftoning filter
# H = np.array([[40, 60, 150, 90, 10], [80, 170, 240, 200, 110], [140, 210, 250, 220, 130], [120, 190, 230, 180, 70], [20, 100, 160, 50, 30]])

# H_height = H.shape[0]
# H_width = H.shape[1]

# for i in range(0, img_grey.shape[0], H_height):
    # for j in range(0, img_grey.shape[1], H_width):

        # img_grey[i:i+H_height,j:j+H_width] = img_grey[i:i+H_height,j:j+H_width] + H
        
        
# # print(img_grey.max())        
        
# # Simple thresholding
# img_grey_bin_thresh = np.zeros(img_grey.shape)
# mask = img_grey>255

# img_grey_bin_thresh[mask] = 1

# # Show thresholded image.
# # Set window with given size to show images.
# cv2.namedWindow("Thresholded image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Thresholded image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Thresholded image', img_grey_bin_thresh)
# cv2.waitKey(0)



# """
# Part 4: Halftoning by ordered dithering, alternate H that maximizes the distance between the dots that get turned on
# """

# # Read image
# img = cv2.imread('Images/parade_lec_20.jpg')

# # Convert image to greyscale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Reshape image to the size which is multiple of 4
# new_width = 1024
# new_height = 1024
# img_grey = cv2.resize(img_grey, (new_width, new_height))

# # Display resized image
# cv2.namedWindow("Resized image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Resized image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Resized image", img_grey.shape[1]*2, img_grey.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('Resized image', img_grey)


# # Convert image to float
# img_grey = img_grey.astype(np.float32)

# # Create halftoning filter
# H = np.array([[1, 9, 3, 10], [13, 5, 15, 7], [4, 12, 2, 10], [16, 8, 14, 6]])*16

# H_height = H.shape[0]
# H_width = H.shape[1]

# print(H_height)

# for i in range(0, img_grey.shape[0], H_height):
    # for j in range(0, img_grey.shape[1], H_width):

        # img_grey[i:i+H_height,j:j+H_width] = img_grey[i:i+H_height,j:j+H_width] + H
        
        
# # print(img_grey.max())        
        
# # Simple thresholding
# img_grey_bin_thresh = np.zeros(img_grey.shape)
# mask = img_grey>255

# img_grey_bin_thresh[mask] = 1

# # Show thresholded image.
# # Set window with given size to show images.
# cv2.namedWindow("Thresholded image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Thresholded image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Resized image", img_grey_bin_thresh.shape[1], img_grey_bin_thresh.shape[0])  # Set window to to twice the size
# cv2.imshow('Thresholded image', img_grey_bin_thresh)
# cv2.waitKey(0)




"""
Part 5:Error diffusion dithering (Floyd-Steinberg ditehring)
"""

# Read image
img = cv2.imread('Images/parade_lec_20.jpg')

# Convert image to greyscale
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display resized image
cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Original image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.resizeWindow("Original image", img_grey.shape[1], img_grey.shape[0])  # Set window to to twice the size
cv2.imshow('Original image', img_grey)


# Convert image to float
img_grey = img_grey.astype(np.float32)

# Set the threshold
thrsh = 128

for i in range(0, img_grey.shape[0]):
    for j in range(0, img_grey.shape[1]):
    
        if img_grey[i,j]<thrsh: 
            err = img_grey[i,j] - 0
        else:
            err = img_grey[i,j] - 255 
        
        try: 
            img_grey[i,j+1] += (7.0/16)*err
        except Exception as e:
            pass            
            
        try: 
            img_grey[i+1,j-1] += (3.0/16)*err
        except Exception as e:
            pass            
            
        try: 
            img_grey[i+1,j] += (5.0/16)*err
        except Exception as e:
            pass            
            
        try: 
            img_grey[i+1,j+1] += (1.0/16)*err
        except Exception as e:
            pass            
        
                       
# Simple thresholding
img_grey_bin_thresh = np.zeros(img_grey.shape)
mask = img_grey>128

img_grey_bin_thresh[mask] = 1

# Show thresholded image.
# Set window with given size to show images.
cv2.namedWindow("Thresholded image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Thresholded image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.resizeWindow("Thresholded image", img_grey_bin_thresh.shape[1], img_grey_bin_thresh.shape[0])  # Set window to to twice the size
cv2.imshow('Thresholded image', img_grey_bin_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


