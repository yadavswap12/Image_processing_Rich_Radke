import numpy as np
import cv2
from scipy.signal import wiener


"""
Part 1: Remove periodic noise by notch filter in frequency domain
"""

# # Read image
# img = cv2.imread('Images/pompeii_periodic_noise_lec_17.jpeg')

# # Convert image to greyscale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# # fft
# img_fft = cv2.dft(np.float32(img_grey), flags=cv2.DFT_REAL_OUTPUT)

# # Shift to zero frequency component 
# img_fft_shift = np.fft.fftshift(img_fft)

# img_fft_shift_display = cv2.normalize(img_fft_shift, None, 0, 255, cv2.NORM_MINMAX)
# img_fft_shift_display = np.uint8(img_fft_shift_display)

# cv2.imshow('image-fft magnitude', img_fft_shift_display)
# # cv2.waitKey(0)


# # Get coordinates of high frequency components from mous click event.

# def click_event(event, x, y, flags, param):
    
    # global x_click, y_click
    
    # if event == cv2.EVENT_LBUTTONDOWN:
        # print(f"Coordinates: ({x}, {y})")
        # x_click, y_click = (y, x)



# # # print(type(img))
# # print(img_fft_shift.shape)

# # Display image in window
# cv2.namedWindow("Image FFT user input", cv2.WINDOW_NORMAL)
# cv2.setMouseCallback("Image FFT user input", click_event)
# print('Click on high intensity point away from center.')
# # cv2.waitKey(0)


# while True:
    # cv2.imshow('Image FFT user input', img_fft_shift_display)
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord('q'):  # Press 'q' to quit and get the contour
        # break

# # cv2.destroyAllWindows()




# # Get the center coordinate of the image-FFT. 
# height, width = img_fft_shift.shape[:2]

# # Calculate the center coordinates
# center_x = height // 2
# center_y = width // 2

# # Get the radius for notch-filter
# r_notch = int(((x_click-center_x)**2 + (y_click-center_y)**2)**(0.5))

# print(f'Radius of notch is {r_notch}')

# # Define notch filter (as circular ring with certain width where the value is zero and value is 1 elsewhere)

# def filter_fft_notch(img, r, r_width):

    # # Get the center coordinate of the image-FFT. 
    # height, width = img.shape[:2]
    
    

    # # Calculate the center coordinates
    # # center_x = height//2
    # # center_y = width//2
    # center_x = width//2
    # center_y = height//2    

    # center = (center_x, center_y)
    
    # print(f'Center is {center}') 


    # F_notch1 = np.zeros((img.shape[0], img.shape[1]))
    # print(f'Shape of f_notch1 is {F_notch1.shape}')
    # F_inner = cv2.circle(F_notch1, center=center, radius=r-r_width//2, color=(1), thickness=cv2.FILLED)
    
    # F_notch2 = (-1)*np.ones((img.shape[0], img.shape[1]))
    # F_outer = cv2.circle(F_notch2, center=center, radius=r+r_width//2, color=(0), thickness=cv2.FILLED)  

    # # Show image
    # cv2.imshow('F_inner', F_inner)
    # cv2.imshow('F_outer', F_outer)    
    
    # F_mask = F_inner-F_outer

    # return F_mask
   
  

# print(f'Size of image fft is {img_fft_shift.shape}') 

# # Create ring-shaped notch filter
# F_fft_abs_shift = filter_fft_notch(img_fft_shift, r=r_notch, r_width=15) 

# # Normalize for display
# F_fft_abs_shift_display = cv2.normalize(F_fft_abs_shift, None, 0, 255, cv2.NORM_MINMAX)
# F_fft_abs_shift_display = np.uint8(F_fft_abs_shift_display) 

# # Show image
# cv2.imshow('F-fft-ring-shaped notch filter', F_fft_abs_shift_display)
# # cv2.waitKey(0)  






# # Apply the notch-filter.

# # Apply filter in fourier domain.
# # filt_img_fft_shift = img_fft_shift*F_fft_abs_shift.reshape(F_fft_abs_shift.shape[0], F_fft_abs_shift.shape[1], 1)    # (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
# filt_img_fft_shift = img_fft_shift*F_fft_abs_shift
# # filt_img_fft = img_fft_abs*F_fft_abs

# # Reverse the shift for image fft.
# filt_img_fft = np.fft.ifftshift(filt_img_fft_shift)

# # Perform Inverse DFT
# # filt_img = cv2.idft(filt_img_fft, flags=cv2.DFT_COMPLEX_OUTPUT)    # (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
# filt_img = cv2.idft(filt_img_fft, flags=cv2.DFT_REAL_OUTPUT)

# # Take the magnitude and normalize for display
# # filt_img = cv2.magnitude(filt_img[:,:,0], filt_img[:,:,1])    # (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
# filt_img = cv2.normalize(filt_img, None, 0, 255, cv2.NORM_MINMAX)
# filt_img = np.uint8(filt_img)


# # Display result


# # Display result
# cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("SIFT", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Original image", img_grey.shape[1]*2, img_grey.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('Original image', img_grey)

# cv2.namedWindow("filtered image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("SIFT", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("filtered image", filt_img.shape[1]*2, filt_img.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('filtered image', filt_img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()








"""
Part 2: Adaptive filtering, Wiener filter
"""

# # Read image
# img = cv2.imread('Images/moon_lec_17.jpeg')

# # Convert image to greyscale
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img_grey = cv2.normalize(img_grey, None, 0, 255, cv2.NORM_MINMAX)
# img_grey = np.uint8(img_grey)


# # Add Gaussian noise to the image.
# mean = 0
# std_dev = 2
# gaussian_noise = np.random.normal(mean, std_dev, img_grey.shape).astype(np.float32)
# # gaussian_noise = np.random.normal(mean, std_dev, img_grey.shape).astype(np.uint8)

# img_noise = img_grey + gaussian_noise

# # Apply the Wiener filter
# # mysize is the size of the Wiener filter window (e.g., 5 for 5x5)
# # If noise is None, it's estimated as the average of the local variance
# filt_img = wiener(img_noise, mysize=2) 

# filt_img = cv2.normalize(filt_img, None, 0, 255, cv2.NORM_MINMAX)
# filt_img = np.uint8(filt_img)

# print(filt_img.max())
# print(filt_img[0,:].max())


# # Display result
# cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("SIFT", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("Original image", img_grey.shape[1]*2, img_grey.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('Original image', img_grey)


# # Show noisy image.
# img_noise = cv2.normalize(img_noise, None, 0, 255, cv2.NORM_MINMAX)
# img_noise = np.uint8(img_noise)

# # Set window with given size to show images.
# cv2.namedWindow("Image with noise", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Image with noise", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.imshow('Image with noise', img_noise)

# cv2.namedWindow("filtered image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("SIFT", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
# cv2.resizeWindow("filtered image", filt_img.shape[1]*2, filt_img.shape[0]*2)  # Set window to to twice the size
# cv2.imshow('filtered image', filt_img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()



"""
Part 3: Wiener filter deconvolution (for removing the blur and additive noise together).
"""

# Read image
img = cv2.imread('Images/moon_with_stars_lec_17.jpg')

# Convert image to greyscale
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Add Gaussian noise to the image.
mean = 0
std_dev = 4
gaussian_noise = np.random.normal(mean, std_dev, img_grey.shape).astype(np.float32)
# gaussian_noise = np.random.normal(mean, std_dev, img_grey.shape).astype(np.uint8)

img_noise = img_grey + gaussian_noise

# Show noisy image.
img_noise_display = cv2.normalize(img_noise, None, 0, 255, cv2.NORM_MINMAX)
img_noise_display = np.uint8(img_noise_display)

# # Choose template for blurred point-spread-function(psf) (for eg. stars) (Only need to run the first time)
# roi = cv2.selectROI("Select template for blurred point-spread-function (for eg. stars)", img_noise_display, fromCenter=False, showCrosshair=True)

# # Extract coordinates, width, and height
# x, y, w, h = roi

# print(roi)

# img_psf_blur = np.zeros(img_noise.shape)

# if w > 0 and h > 0:  # Check if a valid ROI was selected

    # center_x = img_noise.shape[1]//2 
    # center_y = img_noise.shape[0]//2

    # print(img_noise.shape)    
    
    # # Step 3: Crop the image
    # img_psf_blur[center_y-h//2:center_y+h//2+1, center_x-w//2:center_x+w//2+1] = img_noise[y:y+h, x:x+w]

    # # Step 4: Save the cropped image
    # output_path = "Images/img_psf_blur.jpg"  # Desired output path
    # cv2.imwrite(output_path, img_psf_blur)


# Read saved psf
img_psf_blur = cv2.imread('Images/img_psf_blur.jpg')

# Convert image to greyscale
img_psf_blur = cv2.cvtColor(img_psf_blur, cv2.COLOR_BGR2GRAY)

# FFT of image (Real output is fine?)
# img_fft = cv2.dft(np.float32(img_noise), flags=cv2.DFT_REAL_OUTPUT)
img_fft = cv2.dft(np.float32(img_noise), flags=cv2.DFT_COMPLEX_OUTPUT)

# # Shift to zero frequency component (not necessary as we are not displaying) 
# img_fft = np.fft.fftshift(img_fft)

# FFT of psf (need complex version)
H = cv2.dft(np.float32(img_psf_blur), flags=cv2.DFT_COMPLEX_OUTPUT)

print(f'shape of img_fft is {img_fft.shape}')

# # Shift to zero frequency component (not necessary as we are not displaying) 
# H = np.fft.fftshift(H)

# Get deconvolution Wiener filter (H_wiener(u,v) = H_conj(u,v) / (|H(u,v)|^2 + NSR))
NSR = 0.001    # Noise to signal ratio
H_conj = H.conj()
H_wiener = H_conj/(abs(H)**2 + NSR)
# H_wiener = H_conj/(H.real**2 + H.imag**2 + NSR)

# # Calculate magnitude
# H_wiener = cv2.magnitude(H_wiener[:,:,0], H_wiener[:,:,1])    # (use this if flags=cv2.DFT_COMPLEX_OUTPUT)

# # Apply filter in fourier domain.
# filt_img_fft = img_fft*H_wiener.reshape(H_wiener.shape[0], H_wiener.shape[1], 1)    # (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
# filt_img_fft = img_fft*H_wiener
# # filt_img_fft = img_fft_abs*F_fft_abs

# Apply Wiener filter
filt_img_fft = cv2.mulSpectrums(img_fft, H_wiener, 0)

# # Reverse the shift for image fft (not necessary as we are not displaying).
# filt_img_fft = np.fft.ifftshift(filt_img_fft)


# Perform Inverse DFT
# filt_img = cv2.idft(filt_img_fft, flags=cv2.DFT_COMPLEX_OUTPUT)    # (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
filt_img = cv2.idft(filt_img_fft, flags=cv2.DFT_REAL_OUTPUT)
# filt_img = cv2.idft(filt_img_fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)



# Take the magnitude and normalize for display
# filt_img = cv2.magnitude(filt_img[:,:,0], filt_img[:,:,1])    # (use this if flags=cv2.DFT_COMPLEX_OUTPUT)
filt_img = cv2.normalize(filt_img, None, 0, 255, cv2.NORM_MINMAX)
filt_img = np.uint8(filt_img)


# Set window with given size to show images.
cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Original image", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.resizeWindow("Original image", int(img_grey.shape[1]*0.25), int(img_grey.shape[0]*0.25))  # Set window to to twice the size
cv2.imshow('Original image', img_grey)

# Set window with given size to show images.
cv2.namedWindow("Image with noise", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image with noise", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.resizeWindow("Image with noise", int(img_noise_display.shape[1]*0.25), int(img_noise_display.shape[0]*0.25))  # Set window to to twice the size
cv2.imshow('Image with noise', img_noise_display)

cv2.namedWindow("filtered image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("SIFT", 512, 512)  # Set window to 512 pixels wide, 512 pixels high
cv2.resizeWindow("filtered image", int(filt_img.shape[1]*0.25), int(filt_img.shape[0]*0.25))  # Set window to to twice the size
cv2.imshow('filtered image', filt_img)

cv2.waitKey(0)
cv2.destroyAllWindows()






