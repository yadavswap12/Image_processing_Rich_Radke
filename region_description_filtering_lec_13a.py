import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
Part 1: CV2 functions similar to matlab's regionprops

 (e.g., cv2.connectedComponentsWithStats, cv2.minEnclosingCircle, cv2.fitEllipse)

"""

# # Read image
# img = cv2.imread('Images/totem_lec_13a.jpg')


# # Convert image to greyscale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Create binary image by Adaptive-thresholding

# # # Apply adaptive thresholding using ADAPTIVE_THRESH_MEAN_C
# # img_bin = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 105, 0)

# # Apply adaptive thresholding using ADAPTIVE_THRESH_GAUSSIAN_C
# # # img_bin = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 0)
# # img_bin = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 0)    # Inverted binary image

# # Apply Otsu's binarization
# # ret, img_bin = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# ret, img_bin = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)    # Inverted binary image

# # Show thresholded image
# # Set window with given size to show images
# cv2.namedWindow("Adaptive thresholding with Gaussian weighted mean", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Adaptive thresholding with Gaussian weighted mean", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Adaptive thresholding with Gaussian weighted mean", img.shape[1]*2, img.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('Adaptive thresholding with Gaussian weighted mean', img_bin)
# # cv2.waitKey(0)


# # Find connected components and their stats
# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin, connectivity=4)

# # print(labels[2])
# # print(labels.shape)


# # Create area and shape related thresholds for bounding box.
# a_low_lim = 50
# a_up_lim = 10000
# wd_ht_diff_lim = 2

# # Create list for filtered components
# all_component_contours = []

# # Iterate through each labeled component (starting from 1 as 0 is background)
# for i in range(1, num_labels):
    # x, y, w, h, area = stats[i]
    # cx, cy = centroids[i]
    # # if (abs(h*w)>a_low_lim) and (abs(h*w)<=a_up_lim) and (abs(h-w)<=wd_ht_diff_lim):
    # if (area>a_low_lim) and (area<=a_up_lim) and (abs(h-w)<=wd_ht_diff_lim):


        # # Create a mask for the current component
        # component_mask = (labels == i).astype(np.uint8) * 255

        # # Find contours in the component mask
        # contours, hierarchy = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # Add the found contours to a list
        # all_component_contours.extend(contours)
        
        

# for i in range(len(all_component_contours)):
  
    # # Draw all contours in green with a thickness of 2
    # contours_out = all_component_contours[i]
    # cv2.drawContours(img, contours_out, -1, (0, 255, 0), 2) 
  
        


# # # Normalize and convert to unit8.
# # drawing_image = cv2.normalize(drawing_image, None, 0, 255, cv2.NORM_MINMAX)
# # drawing_image = np.uint8(drawing_image)

# # Show contours image.
# # Set window with given size to show images.
# cv2.namedWindow("Filtered regions image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Filtered regions image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Filtered regions image", img.shape[1]*2, img.shape[0]*2)  # Set window to twice the size
# cv2.imshow('Filtered regions image', img)
# cv2.waitKey(0)        
        
        
        
        
"""
Part 2: Skeletons
"""        
 
# # Read image
# img = cv2.imread('Images/A_lec_13a.jpg')


# # Convert image to greyscale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Create binary image by Adaptive-thresholding

# # # Apply adaptive thresholding using ADAPTIVE_THRESH_MEAN_C
# # img_bin = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 105, 0)

# # Apply adaptive thresholding using ADAPTIVE_THRESH_GAUSSIAN_C
# # # img_bin = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 0)
# # img_bin = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 0)    # Inverted binary image

# # Apply Otsu's binarization
# # ret, img_bin = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# ret, img_bin = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)    # Inverted binary image

# # Show thresholded image
# # Set window with given size to show images
# cv2.namedWindow("Adaptive thresholding with Gaussian weighted mean", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Adaptive thresholding with Gaussian weighted mean", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Adaptive thresholding with Gaussian weighted mean", img.shape[1]*2, img.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('Adaptive thresholding with Gaussian weighted mean', img_bin)
# # cv2.waitKey(0)


# # Create structuring element
# # # Simple structuring element
# # kernel = np.ones((3,3), np.uint8)
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2))

# # Apply erosion
# dilated_image = cv2.dilate(img_bin, kernel, iterations=1)

# # Show eroded image
# # Set window with given size to show images
# cv2.namedWindow("Dilated image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Dilated image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Dilated image", img.shape[1]*2, img.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('Dilated image', dilated_image)
# # cv2.waitKey(0)

 
# # Perform skeletonization (thinning)
# img_skeleton = cv2.ximgproc.thinning(dilated_image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

# # Show skeleton image
# # Set window with given size to show images
# cv2.namedWindow("Skeleton image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Skeleton image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Skeleton image", img.shape[1]*2, img.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('Skeleton image', img_skeleton)
# # cv2.waitKey(0)

# cv2.waitKey(0)



"""
Part 3: Grey level co-occurance matrix
"""

from skimage.feature import graycomatrix, graycoprops

# Read image
img1 = cv2.imread('Images/wheat_thin_lec_13a.jpg')
img2 = cv2.imread('Images/triscuit_lec_13a.jpg')

# Show original image
# Set window with given size to show images
cv2.namedWindow("original image 1", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("original image 1", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.resizeWindow("original image 1", img1.shape[1]*2, img1.shape[0]*2)  # Set window to to twice the size
cv2.imshow('original image 1', img1)
# cv2.waitKey(0)

# Show original image
# Set window with given size to show images
cv2.namedWindow("original image 2", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("original image 2", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.resizeWindow("original image 2", img2.shape[1]*2, img2.shape[0]*2)  # Set window to to twice the size
cv2.imshow('original image 2', img2)
# cv2.waitKey(0)

cv2.waitKey(0)


# Convert image to greyscale
img_grey1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img_grey2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 0, 45, 90, 135 degrees
levels = 256 # Assuming 8-bit grayscale image

pix_dist_list = []
corr1_mean_list = []
corr2_mean_list = []
corr1_hor_list = []
corr2_hor_list = []

# Get correlations for different distances
for i in range(20):
    distances = [i+1]
    pix_dist_list.append(i+1)

    glcm1 = graycomatrix(img_grey1, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    glcm2 = graycomatrix(img_grey2, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    
    # Properties
    contrast1 = graycoprops(glcm1, 'contrast')
    contrast2 = graycoprops(glcm2, 'contrast')
    
    dissimilarity1 = graycoprops(glcm1, 'dissimilarity')
    dissimilarity2 = graycoprops(glcm2, 'dissimilarity')
    
    homogeneity1 = graycoprops(glcm1, 'homogeneity')
    homogeneity2 = graycoprops(glcm2, 'homogeneity')
    
    energy1 = graycoprops(glcm1, 'energy')
    energy2 = graycoprops(glcm2, 'energy')
    
    correlation1 = graycoprops(glcm1, 'correlation')
    # Get mean of correlations for all four directions.
    corr1_mean = np.mean(np.array(correlation1)) 
    # Get correlation for forward-horizontal direction.
    corr1_hor = correlation1[0][0]
    
    corr1_mean_list.append(corr1_mean)
    corr1_hor_list.append(corr1_hor)
    
    
    correlation2 = graycoprops(glcm2, 'correlation')
    # Get mean of correlations for all four directions.
    corr2_mean = np.mean(np.array(correlation2))
    # Get correlation for forward-horizontal direction.
    corr2_hor = correlation2[0][0]    
    
    corr2_mean_list.append(corr2_mean)
    corr2_hor_list.append(corr2_hor)
     
    
# plt.plot(pix_dist_list,corr1_mean_list, label='correlation_wheat_thin')
# # plt.title('correlation_wheat_thin')
# # plt.show()    
    
# plt.plot(pix_dist_list,corr2_mean_list, label='correlation_triscuit')
# # plt.title('correlation_triscuit')
# # plt.show()   

# plt.legend()
# plt.show()   


plt.plot(pix_dist_list,corr1_hor_list, label='correlation_wheat_thin')
# plt.title('correlation_wheat_thin')
# plt.show()    
    
plt.plot(pix_dist_list,corr2_hor_list, label='correlation_triscuit')
# plt.title('correlation_triscuit')
# plt.show()   

plt.legend()
plt.show() 
        
    
    
    


    