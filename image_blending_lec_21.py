import numpy as np
import cv2

"""
Part 1: Binary mask based
"""

# # Read target image
# img = cv2.imread('Images/cow_target_lec_21.jpg')

# # # Convert image to greyscale
# # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Read source image
# img_src = cv2.imread('Images/cow_source_lec_21.jpg')


# # Make sure the source and target images are of same size and are np.int32 (for compatible addition)
# img_width = 512
# img_height = 512

# img_src = cv2.resize(img_src, (img_width, img_height)).astype(np.int32)
# img = cv2.resize(img, (img_width, img_height)).astype(np.int32)

# img_src_copy = img_src.copy().astype(np.uint8)

# # # Convert image to greyscale
# # img_src_grey = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

# # Create a mask for object in source image
# mask = np.zeros(img_src.shape[:2], dtype="uint8")

# # # Define the coordinates for your polygon ROI (example)
# # # You'd typically get these coordinates from user interaction or some other detection method
# # polygon_points = np.array([[100, 100], [200, 50], [300, 100], [250, 200], [150, 200]], np.int32)
# # # polygon_points = polygon_points.reshape((-1, 1, 2))



# # For user drawn ROI contour
# points = []
# drawing = False

# def draw_contour(event, x, y, flags, param):
    # global points, drawing, img_src

    # if event == cv2.EVENT_LBUTTONDOWN:
        # drawing = True
        # points.append((x, y))
    # elif event == cv2.EVENT_MOUSEMOVE:
        # if drawing:
            # points.append((x, y))
            # # # Optional: Draw on a copy of the image for visual feedback
            # cv2.circle(img_src_copy, (x, y), 2, (0, 0, 255), -1)
            # cv2.imshow("Image", img_src_copy)
    # elif event == cv2.EVENT_LBUTTONUP:
        # drawing = False


# # Show Original image.
# # Set window with given size to show images.
# cv2.namedWindow("Source image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Source image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.setMouseCallback("Source image", draw_contour)
# print('Select initial conotur around object to be segmented.')
# # cv2.waitKey(0)


# while True:
    # cv2.imshow("Source image", img_src.astype(np.uint8))
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord('q'):  # Press 'q' to quit and get the contour
        # break


# # init = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
# # init = np.array(points, dtype=np.int32).reshape((1, 2, -1))
# cntr_pts = np.array(points, dtype=np.int32)
# # Switch x,y as the order in mouse-event is reveresed 
# # cntr_pts[:,0] = np.array(points, dtype=np.int32)[:, 1]
# # cntr_pts[:,1] = np.array(points, dtype=np.int32)[:, 0]
# cntr_pts = cntr_pts.reshape((-1, 1, 2))




# # Draw the polygon on the mask, filling it with white (255)
# # cv2.fillPoly(mask, [polygon_points], 255)
# cv2.fillPoly(mask, [cntr_pts], 255)

# # Apply the mask to the original image
# masked_image = cv2.bitwise_and(img_src, img_src, mask=mask)

# # Show masked image.
# # Set window with given size to show images.
# cv2.namedWindow("masked image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("masked image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("masked image", masked_image.shape[1]*2, masked_image.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('masked image', masked_image.astype(np.uint8))
# # cv2.waitKey(0)


# # Blend the images together
# img += masked_image

# # Scale blended image
# img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
# img = np.uint8(img)

# # Show blended image.
# # Set window with given size to show images.
# cv2.namedWindow("blended image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("blended image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("blended image", img.shape[1]*1, img.shape[0]*1)  # Set window to to twice the size
# cv2.imshow('blended image', img.astype(np.uint8))
# cv2.waitKey(0)



"""
Part 2: Smoothened binary mask based
"""

# # Read target image
# img = cv2.imread('Images/cow_target_lec_21.jpg')

# # # Convert image to greyscale
# # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Read source image
# img_src = cv2.imread('Images/cow_source_lec_21.jpg')


# # Make sure the source and target images are of same size and are np.int32 (for compatible addition)
# img_width = 512
# img_height = 512

# img_src = cv2.resize(img_src, (img_width, img_height)).astype(np.int32)
# img = cv2.resize(img, (img_width, img_height)).astype(np.int32)

# img_src_copy = img_src.copy().astype(np.uint8)

# # # Convert image to greyscale
# # img_src_grey = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

# # Create a mask for object in source image
# mask = np.zeros(img_src.shape[:2], dtype="uint8")

# # # Define the coordinates for your polygon ROI (example)
# # # You'd typically get these coordinates from user interaction or some other detection method
# # polygon_points = np.array([[100, 100], [200, 50], [300, 100], [250, 200], [150, 200]], np.int32)
# # # polygon_points = polygon_points.reshape((-1, 1, 2))



# # For user drawn ROI contour
# points = []
# drawing = False

# def draw_contour(event, x, y, flags, param):
    # global points, drawing, img_src

    # if event == cv2.EVENT_LBUTTONDOWN:
        # drawing = True
        # points.append((x, y))
    # elif event == cv2.EVENT_MOUSEMOVE:
        # if drawing:
            # points.append((x, y))
            # # # Optional: Draw on a copy of the image for visual feedback
            # cv2.circle(img_src_copy, (x, y), 2, (0, 0, 255), -1)
            # cv2.imshow("Image", img_src_copy)
    # elif event == cv2.EVENT_LBUTTONUP:
        # drawing = False


# # Show Original image.
# # Set window with given size to show images.
# cv2.namedWindow("Source image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Source image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.setMouseCallback("Source image", draw_contour)
# print('Select initial conotur around object to be segmented.')
# # cv2.waitKey(0)


# while True:
    # cv2.imshow("Source image", img_src.astype(np.uint8))
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord('q'):  # Press 'q' to quit and get the contour
        # break


# # init = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
# # init = np.array(points, dtype=np.int32).reshape((1, 2, -1))
# cntr_pts = np.array(points, dtype=np.int32)
# # Switch x,y as the order in mouse-event is reveresed 
# # cntr_pts[:,0] = np.array(points, dtype=np.int32)[:, 1]
# # cntr_pts[:,1] = np.array(points, dtype=np.int32)[:, 0]
# cntr_pts = cntr_pts.reshape((-1, 1, 2))




# # Draw the polygon on the mask, filling it with white (255)
# # cv2.fillPoly(mask, [polygon_points], 255)
# # cv2.fillPoly(mask, [cntr_pts], 255)
# cv2.fillPoly(mask, [cntr_pts], 1)

# # Smoothened mask
# mask = mask.astype(np.float32)

# # # 2D spatial filter: smoothing (low pass filter)
# # # F = (1.0/9)*np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
# # F = (1.0/36)*np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])

# # # Apply filter.
# # mask = cv2.filter2D(mask, ddepth=-1, kernel=F, dst=None, anchor=None, delta=None, borderType=None)

# # Apply Gaussian blurring
# kernel_size = (5, 5) # Kernel size (must be odd and positive)
# sigma_x = 0 # Sigma in X direction (0 means calculated from kernel size)
# mask = cv2.GaussianBlur(mask, kernel_size, sigma_x)

# # # Apply the mask to the original image
# # masked_image = cv2.bitwise_and(img_src, img_src, mask=mask)

# # # Show masked image.
# # # Set window with given size to show images.
# # cv2.namedWindow("masked image", cv2.WINDOW_NORMAL)
# # # cv2.resizeWindow("masked image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# # cv2.resizeWindow("masked image", masked_image.shape[1]*2, masked_image.shape[0]*2)  # Set window to to twice the size
# # cv2.imshow('masked image', masked_image.astype(np.uint8))
# # # cv2.waitKey(0)


# # Reshape mask for color image
# mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)


# # Blend the images together
# # img += masked_image
# img = img*(1-mask) + img_src*mask

# # Scale blended image
# img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
# img = np.uint8(img)

# # Show blended image.
# # Set window with given size to show images.
# cv2.namedWindow("blended image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("blended image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("blended image", img.shape[1]*1, img.shape[0]*1)  # Set window to to twice the size
# cv2.imshow('blended image', img.astype(np.uint8))
# cv2.waitKey(0)





"""
Part 3: Laplacian pyramid based
"""

# # Read target image
# img = cv2.imread('Images/cow_target_lec_21.jpg')

# # # Convert image to greyscale
# # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Read source image
# img_src = cv2.imread('Images/cow_source_lec_21.jpg')


# # Make sure the source and target images are of same size and are np.int32 (for compatible addition)
# img_width = 512
# img_height = 512

# img_src = cv2.resize(img_src, (img_width, img_height)).astype(np.int32)
# img = cv2.resize(img, (img_width, img_height)).astype(np.int32)

# img_src_copy = img_src.copy().astype(np.uint8)

# # # Convert image to greyscale
# # img_src_grey = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

# # Create a mask for object in source image
# mask = np.zeros(img_src.shape[:2], dtype="uint8")

# # For user drawn ROI contour
# points = []
# drawing = False

# def draw_contour(event, x, y, flags, param):
    # global points, drawing, img_src

    # if event == cv2.EVENT_LBUTTONDOWN:
        # drawing = True
        # points.append((x, y))
    # elif event == cv2.EVENT_MOUSEMOVE:
        # if drawing:
            # points.append((x, y))
            # # # Optional: Draw on a copy of the image for visual feedback
            # cv2.circle(img_src_copy, (x, y), 2, (0, 0, 255), -1)
            # cv2.imshow("Image", img_src_copy)
    # elif event == cv2.EVENT_LBUTTONUP:
        # drawing = False


# # Show Original image.
# # Set window with given size to show images.
# cv2.namedWindow("Source image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Source image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.setMouseCallback("Source image", draw_contour)
# print('Select initial conotur around object to be segmented.')
# # cv2.waitKey(0)


# while True:
    # cv2.imshow("Source image", img_src.astype(np.uint8))
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord('q'):  # Press 'q' to quit and get the contour
        # break



# cntr_pts = np.array(points, dtype=np.int32)
# # Switch x,y as the order in mouse-event is reveresed 
# cntr_pts = cntr_pts.reshape((-1, 1, 2))


# # Draw the polygon on the mask, filling it with white (255)
# # cv2.fillPoly(mask, [polygon_points], 255)
# # cv2.fillPoly(mask, [cntr_pts], 255)
# cv2.fillPoly(mask, [cntr_pts], 1)



# # 3-channel mask
# mask = mask.astype(np.float32)

# # Reshape mask for color image
# mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)


# # Create Gaussian pyramid

# def gaussian_pyramid(img_inp, levels):

    # # Convert to compatible format
    # img_inp = img_inp.astype(np.float32)

    # # Initialize the pyramid list with original image
    # gauss_pyr_list = [img_inp]

    # for i in range(1, levels):
        # img_dwn_smpl = cv2.pyrDown(img_inp)
        # gauss_pyr_list.append(img_dwn_smpl)
        # img_inp = img_dwn_smpl.copy()

    # return gauss_pyr_list



# # Create Gaussian pyramid lists for target, source and mask
# # Select no. of levels
# pyr_level = 4

# img_trg_g_pyr_list = gaussian_pyramid(img, pyr_level) 
# img_src_g_pyr_list = gaussian_pyramid(img_src, pyr_level)     
# mask_g_pyr_list = gaussian_pyramid(mask, pyr_level)   


# # Create Laplacian pyramid lists

# def laplacian_pyramid(img_pyr_list):

    # # Initialize the pyramid list with original image
    # lplc_pyr_list = []  

    # for i in range(len(img_pyr_list)):
    
        # if i == len(img_pyr_list)-1:
        
            # img_i = img_pyr_list[i]
            # lplc_img = img_i       
        
        # else:        
    
            # img_i = img_pyr_list[i]
            # img_i_1 = cv2.pyrUp(img_pyr_list[i+1])
            # lplc_img = img_i - img_i_1
        
        # lplc_pyr_list.append(lplc_img)

    # return lplc_pyr_list    

# # Create laplacian pyramid lists for target, source and mask
# img_trg_l_pyr_list = laplacian_pyramid(img_trg_g_pyr_list)  
# img_src_l_pyr_list = laplacian_pyramid(img_src_g_pyr_list)  
# mask_l_pyr_list = laplacian_pyramid(mask_g_pyr_list)



# # Blend pyramids

# def blend_laplacian_pyramids(img_l_pyr_list, img_src_l_pyr_list, mask_l_pyr_list):

    # # Initialize the pyramid list with original image
    # blnd_lplc_pyr_list = []      

    # for img_l, img_src_l, mask_l in zip(img_l_pyr_list, img_src_l_pyr_list, mask_l_pyr_list):
        
        # img_blend_l = img_l*(1-mask_l) + img_src_l*mask_l
        
        # blnd_lplc_pyr_list.append(img_blend_l)
        
    # return blnd_lplc_pyr_list    

        
# blnd_l_pyr_list = blend_laplacian_pyramids(img_trg_l_pyr_list, img_src_l_pyr_list, mask_l_pyr_list)
        
        
# # Reconstruct the blended-image from blended-laplacian image

# def reconstruct_blended_laplacian(blnd_l_pyr_list):

    # # blnd_g_pyr_list = []
    
    # for i in range(len(blnd_l_pyr_list)):
        # k = len(blnd_l_pyr_list) - 1 - i
    
        # if i==0:   
            # img_g_k = blnd_l_pyr_list[k]
                    
        # else:
            # img_g_k = blnd_l_pyr_list[k] + cv2.pyrUp(img_g_k)            
            
    # return img_g_k    

  
# img_blnd = reconstruct_blended_laplacian(blnd_l_pyr_list)   
  
# # Show blended image.
# # Set window with given size to show images.
# cv2.namedWindow("blended image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("blended image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("blended image", img_blnd.shape[1]*1, img_blnd.shape[0]*1)  # Set window to to twice the size
# cv2.imshow('blended image', img_blnd.astype(np.uint8))
# cv2.waitKey(0)




"""
Part 3: Poisson image editing
"""

# Read target image
img = cv2.imread('Images/hand_lec_21.jpg')

# # Convert image to greyscale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Read source image
img_src = cv2.imread('Images/face_lec_21.jpg')


# Make sure the source and target images are of same size and are np.int32 (for compatible addition)
img_width = 512
img_height = 512

# img_src = cv2.resize(img_src, (img_width, img_height)).astype(np.int32)
# img = cv2.resize(img, (img_width, img_height)).astype(np.int32)
img_src = cv2.resize(img_src, (img_width, img_height))
img = cv2.resize(img, (img_width, img_height))

img_src_copy = img_src.copy().astype(np.uint8)

# # Convert image to greyscale
# img_src_grey = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

# Create a mask for object in source image
mask = np.zeros(img_src.shape[:2], dtype="uint8")

# For user drawn ROI contour
points = []
drawing = False

def draw_contour(event, x, y, flags, param):
    global points, drawing, img_src

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points.append((x, y))
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))
            # # Optional: Draw on a copy of the image for visual feedback
            cv2.circle(img_src_copy, (x, y), 2, (0, 0, 255), -1)
            cv2.imshow("Image", img_src_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


# Show Original image.
# Set window with given size to show images.
cv2.namedWindow("Source image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Source image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.setMouseCallback("Source image", draw_contour)
print('Select initial conotur around object to be segmented.')
# cv2.waitKey(0)


while True:
    cv2.imshow("Source image", img_src.astype(np.uint8))
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

# Reshape mask for color image
mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

# Define the center of the region in the target image where the source object will be placed
center_x = img.shape[1] // 2
center_y = img.shape[0] // 2
center_point = (center_x, center_y)

# Perform seamless cloning
# cv2.NORMAL_CLONE for standard Poisson blending
# cv2.MIXED_CLONE for mixed gradients (can be useful for texture transfer)
# cv2.MONOCHROME_TRANSFER for transferring color statistics
img_blnd = cv2.seamlessClone(img_src, img, mask, center_point, cv2.MIXED_CLONE)

# Show blended image.
# Set window with given size to show images.
cv2.namedWindow("blended image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("blended image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.resizeWindow("blended image", img_blnd.shape[1]*1, img_blnd.shape[0]*1)  # Set window to to twice the size
cv2.imshow('blended image', img_blnd.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()








             
        
            

