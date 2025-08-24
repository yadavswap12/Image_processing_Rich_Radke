import numpy as np
import cv2
import matplotlib.pyplot as plt



# """
# Part 1: Original image and its DFT 
# """

# # Read image
# # img = cv2.imread('Images/moon_fft_lec_8.jfif')
# # img = cv2.imread('Images/valley_fft_lec_8.jfif')
# img = cv2.imread('Images/valley2_fft_lec_8.jfif')

# # Convert image to greyscale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # # For testing only. Resize the image
# # img_grey = cv2.resize(img_grey, (20, 20))

# # Show image
# cv2.imshow('original image', img_grey)
# cv2.waitKey(0)

# # fft
# # img_fft = cv2.dft(np.float32(img_grey), flags=cv2.DFT_COMPLEX_OUTPUT)
# img_fft = cv2.dft(np.float32(img_grey), flags=cv2.DFT_REAL_OUTPUT)

# # # Calculate magnitude spectrum for visualization (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
# # img_fft_abs_log = 10*np.log(cv2.magnitude(img_fft[:,:,0], img_fft[:,:,1]))
# # img_fft_abs = cv2.magnitude(img_fft[:,:,0], img_fft[:,:,1])

# # # zero-out DC component value for easy visualization (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
# # img_fft_abs[0][0] = 0

# # # Get the histogram (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
# # plt.hist(img_fft_abs.ravel(), bins=30, color='skyblue', edgecolor='black')
# # plt.show()

# # # Clip the values in the array to the specified range (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
# # img_fft_abs = np.clip(img_fft_abs, 0, 50000)

# # # Get the histogram (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
# # plt.hist(img_fft_abs.ravel(), bins=30, color='skyblue', edgecolor='black')
# # plt.show()

# # Shift to zero frequency component 
# # img_fft_abs_log_shift = np.fft.fftshift(img_fft_abs_log)    # (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
# # img_fft_abs_shift = np.fft.fftshift(img_fft_abs)    # (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
# img_fft_shift = np.fft.fftshift(img_fft)


# # Take the magnitude and normalize for display (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
# # img_fft_shift_display = cv2.magnitude(img_fft_shift[:,:,0], img_fft_shift[:,:,1])
# # img_fft_shift_display = cv2.normalize(img_fft_shift_display, None, 0, 255, cv2.NORM_MINMAX)
# # img_fft_shift_display = np.uint8(img_fft_shift_display)

# img_fft_shift_display = cv2.normalize(img_fft_shift, None, 0, 255, cv2.NORM_MINMAX)
# img_fft_shift_display = np.uint8(img_fft_shift_display)


# # # Normalize for display (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
# # img_fft_abs_log_shift_display = cv2.normalize(img_fft_abs_log_shift, None, 0, 255, cv2.NORM_MINMAX)
# # img_fft_abs_log_shift_display = np.uint8(img_fft_abs_log_shift_display)

# # img_fft_abs_shift_display = cv2.normalize(img_fft_abs_shift, None, 0, 255, cv2.NORM_MINMAX)
# # img_fft_abs_shift_display = np.uint8(img_fft_abs_shift_display)

# # # Show image (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
# # cv2.imshow('image-fft-log-abs', img_fft_abs_log_shift_display)
# # cv2.waitKey(0)
# # cv2.imshow('image-fft-abs', img_fft_abs_shift_display)
# # cv2.waitKey(0)

# cv2.imshow('image-fft magnitude', img_fft_shift_display)
# cv2.waitKey(0)










"""
Part 2: Fourier domain filter (shifted such that zero frequecy is at center)
"""

def filter_fft(img, size, filter_shape, filter_type):

    if filter_type == 'low_pass':
        F_fft = np.zeros((img.shape[0], img.shape[1]))
        
        if filter_shape != 'circular':
            start_x = F_fft.shape[0]//2 - size//2
            end_x = F_fft.shape[0]//2 + size//2 + 1
            start_y = F_fft.shape[1]//2 - size//2
            end_y = F_fft.shape[1]//2 + size//2 + 1
            
            F_fft[start_x:end_x, start_y:end_y] = 1

    return F_fft
   
# # # Read image
# # img = cv2.imread('Images/valley2_fft_lec_8.jfif')

# # # Convert image to greyscale
# # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   

# # print(f'Size of original image is {img_grey.shape}') 

# # # Create square-shaped low-pass filter
# # F_fft_abs_shift = filter_fft(img_grey, size=50, filter_shape=None, filter_type='low_pass') 

# # print(f'Size of fft filter is {F_fft_abs_shift.shape}') 


# # # Normalize for display
# # F_fft_abs_shift_display = cv2.normalize(F_fft_abs_shift, None, 0, 255, cv2.NORM_MINMAX)
# # F_fft_abs_shift_display = np.uint8(F_fft_abs_shift_display) 

# # # Show image
# # cv2.imshow('F-fft-low-pass-square', F_fft_abs_shift_display)
# # cv2.waitKey(0)   










# """
# Part 3: Spatial filter and its fourier transform
# """


# # # 2D spatial filter: low pass (Gaussian)
# # # Get a Gaussian kernel
# # dim_x = img_fft.shape[0]
# # dim_y = img_fft.shape[1]
# # kernel_1d_x = cv2.getGaussianKernel(dim_x, 2)
# # kernel_1d_y = cv2.getGaussianKernel(dim_y, 2)


# # # To create a 2D Gaussian kernel (e.g., for direct convolution)
# # # This involves multiplying the 1D kernel with its transpose
# # # F = np.outer(kernel_1d, kernel_1d.transpose())
# # F = np.outer(kernel_1d_x, kernel_1d_y)



# # 2D spatial filter: High pass (Identity-Low pass)
# # Get Low pass filter

# def filter_spatial(img, size, filter_type):

    # if filter_type == 'low_pass':
        # F = np.zeros((img.shape[0], img.shape[1]))
        
        # start_x = F.shape[0]//2 - size//2
        # end_x = F.shape[0]//2 + size//2 + 1
        # start_y = F.shape[1]//2 - size//2
        # end_y = F.shape[1]//2 + size//2 + 1
        
        # F[start_x:end_x, start_y:end_y] = 1.0/(size**2)
        
    # elif filter_type == 'high_pass':
        # F = np.zeros((img.shape[0], img.shape[1]))
        
        # start_x = F.shape[0]//2 - size//2
        # end_x = F.shape[0]//2 + size//2 + 1
        # start_y = F.shape[1]//2 - size//2
        # end_y = F.shape[1]//2 + size//2 + 1
        
        # F[start_x:end_x, start_y:end_y] = 1.0/(size**2)
        
        # # Get identity filter.
        # I = np.zeros((img.shape[0], img.shape[1]))
        # I[I.shape[0]//2][I.shape[1]//2]=1
        
        # # # For testing only
        # # print(I)

        # F=I-F        
        
    # return F


# # Create high_pass filter
# F = filter_spatial(img_grey, size=5, filter_type='high_pass') 
# # F = filter_spatial(img_grey, size=15, filter_type='low_pass') 


# # Show image
# cv2.imshow('filter', F)
# cv2.waitKey(0)

# # print(F)


# # fft of filter
# # F_fft_abs = cv2.dft(np.float32(F), flags=cv2.DFT_REAL_OUTPUT)
# F_fft = cv2.dft(np.float32(F), flags=cv2.DFT_COMPLEX_OUTPUT)    # (use this if flags=cv2.DFT_COMPLEX_OUTPUT)

# # Calculate magnitude
# F_fft_abs = cv2.magnitude(F_fft[:,:,0], F_fft[:,:,1])    # (use this if flags=cv2.DFT_COMPLEX_OUTPUT)


# # Shift to zero frequency component
# F_fft_abs_shift = np.fft.fftshift(F_fft_abs)

# # Perform Inverse DFT
# # img_back = cv2.idft(F_fft_abs_shift, flags=cv2.DFT_REAL_OUTPUT)
# img_back = cv2.idft(F_fft_abs_shift, flags=cv2.DFT_COMPLEX_OUTPUT)    # (use this if flags=cv2.DFT_COMPLEX_OUTPUT)



# # Take the magnitude and normalize for display
# # img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])    # (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
# img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
# img_back = np.uint8(img_back)


# # Show image
# cv2.imshow('Inverse fft of filter', img_back)
# cv2.waitKey(0)


# # # Calculate magnitude spectrum for visualization
# # F_fft_abs = 10*np.log(cv2.magnitude(F_fft[:,:,0], F_fft[:,:,1]))
# # # F_fft_abs = cv2.magnitude(F_fft[:,:,0], F_fft[:,:,1])

# # # Shift to zero frequency component
# # F_fft_abs_shift = np.fft.fftshift(F_fft_abs)

# # # normalize for display
# # F_fft_abs = cv2.normalize(F_fft_abs, None, 0, 255, cv2.NORM_MINMAX)
# # F_fft_abs = np.uint8(F_fft_abs)

# # normalize for display
# F_fft_abs_shift_display = cv2.normalize(F_fft_abs_shift, None, 0, 255, cv2.NORM_MINMAX)
# F_fft_abs_shift_display = np.uint8(F_fft_abs_shift_display)


# # Show image
# # cv2.imshow('filter fft magnitude', F_fft_abs)
# # cv2.waitKey(0)
# cv2.imshow('filter fft magnitude', F_fft_abs_shift_display)
# cv2.waitKey(0)













# """
# Part 4: Application of filter in Fourier domain
# """


# # Low-pass-filtering using fft.

# # Apply filter in fourier domain.
# # filt_img_fft_shift = img_fft_shift*F_fft_abs_shift.reshape(F_fft_abs_shift.shape[0], F_fft_abs_shift.shape[1], 1)    # (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
# filt_img_fft_shift = img_fft_shift*F_fft_abs_shift
# # filt_img_fft = img_fft_abs*F_fft_abs


# # # normalize for display
# # filt_img_fft_abs_shift_display = cv2.normalize(filt_img_fft_abs_shift, None, 0, 255, cv2.NORM_MINMAX)
# # filt_img_fft_abs_shift_display = np.uint8(filt_img_fft_abs_shift_display)

# # # Show image
# # cv2.imshow('filtered image fft', filt_img_fft_abs_shift_display)
# # cv2.waitKey(0)


# # Reverse the shift for image fft.
# filt_img_fft = np.fft.ifftshift(filt_img_fft_shift)


# # Perform Inverse DFT
# # filt_img = cv2.idft(filt_img_fft, flags=cv2.DFT_COMPLEX_OUTPUT)    # (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
# filt_img = cv2.idft(filt_img_fft, flags=cv2.DFT_REAL_OUTPUT)


# # Take the magnitude and normalize for display
# # filt_img = cv2.magnitude(filt_img[:,:,0], filt_img[:,:,1])    # (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
# filt_img = cv2.normalize(filt_img, None, 0, 255, cv2.NORM_MINMAX)
# filt_img = np.uint8(filt_img)


# # Show image
# cv2.imshow('filtered image', filt_img)
# cv2.waitKey(0)






"""
Part 5: Aliasing
"""




# Read image
# img = cv2.imread('Images/striped_pants_aliasing_lec_8.jfif')
# img = cv2.imread('Images/striped_pants_2_aliasing_lec_8.jfif')
img = cv2.imread('Images/striped_pants_3_aliasing_lec_8.jpg')

# Convert image to greyscale
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Show image
# Set window with given size to show images.
cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.imshow('Original image', img_grey)
cv2.waitKey(0)

# Downsample the image.
img_sampled = img_grey[::4, ::4]

# Show image
# Set window to show images.
cv2.namedWindow("Down-sampled image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Down-sampled image", 512, 512)  # Set window to 512 pixels wide, 512 pixels highcv2.waitKey(0)
cv2.imshow('Down-sampled image', img_sampled)
cv2.waitKey(0)

# Low pass filter in frequency domain
# Create square-shaped low-pass filter
F_fft_abs_shift = filter_fft(img_grey, size=300, filter_shape=None, filter_type='low_pass') 

# fft for image
# img_fft = cv2.dft(np.float32(img_grey), flags=cv2.DFT_COMPLEX_OUTPUT)
img_fft = cv2.dft(np.float32(img_grey), flags=cv2.DFT_REAL_OUTPUT)

# Shift to zero frequency component 
# img_fft_abs_log_shift = np.fft.fftshift(img_fft_abs_log)    # (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
# img_fft_abs_shift = np.fft.fftshift(img_fft_abs)    # (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
img_fft_shift = np.fft.fftshift(img_fft)


# Apply filter in fourier domain.
# filt_img_fft_shift = img_fft_shift*F_fft_abs_shift.reshape(F_fft_abs_shift.shape[0], F_fft_abs_shift.shape[1], 1)    # (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
filt_img_fft_shift = img_fft_shift*F_fft_abs_shift
# filt_img_fft = img_fft_abs*F_fft_abs


# Reverse the shift for image fft.
filt_img_fft = np.fft.ifftshift(filt_img_fft_shift)


# Perform Inverse DFT
# filt_img = cv2.idft(filt_img_fft, flags=cv2.DFT_COMPLEX_OUTPUT)    # (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
filt_img = cv2.idft(filt_img_fft, flags=cv2.DFT_REAL_OUTPUT)


# Take the magnitude and normalize for display
# filt_img = cv2.magnitude(filt_img[:,:,0], filt_img[:,:,1])    # (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
filt_img = cv2.normalize(filt_img, None, 0, 255, cv2.NORM_MINMAX)
filt_img = np.uint8(filt_img)


# Show image
# Set window to show images.
cv2.namedWindow("filtered image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("filtered image", 512, 512)  # Set window to 512 pixels wide, 512 pixels highcv2.waitKey(0)
cv2.imshow('filtered image', filt_img)
cv2.waitKey(0)

# Downsample the filtered image.
img_filt_sampled = filt_img[::4, ::4]

# Show image
# Set window to show images.
cv2.namedWindow("Down-sampled image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Down-sampled image", 512, 512)  # Set window to 512 pixels wide, 512 pixels highcv2.waitKey(0)
cv2.imshow('Down-sampled image', img_filt_sampled)
cv2.waitKey(0)





