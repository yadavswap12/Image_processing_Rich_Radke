import numpy as np
import cv2

# # Read image
# img = cv2.imread('Images/hut_fft_lec_7.jfif')

# # Convert to greyscale 
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Show original image
# cv2.imshow('original image', img_grey)
# cv2.waitKey(0)

# # fft
# img_fft = cv2.dft(np.float32(img_grey), flags=cv2.DFT_COMPLEX_OUTPUT)

# # Shift the zero frequency component
# img_fft_shift = np.fft.fftshift(img_fft)

# # Calculate magnitude spectrum for visualization
# img_fft_shift_abs = 10*np.log(cv2.magnitude(img_fft_shift[:,:,0], img_fft_shift[:,:,1]))

# # This will truncate the decimal part and clip values outside [0, 255]
# img_fft_shift_abs = img_fft_shift_abs.astype(np.uint8)

# # Show fft magnitiude image
# cv2.imshow('fft magnitude', img_fft_shift_abs)
# cv2.waitKey(0)


# # Get the zero frequency component magnitude.
# zero_freq_mag = cv2.magnitude(img_fft[:,:,0], img_fft[:,:,1])[0][0]
# print(f'zero frequency component maginitude is {zero_freq_mag}')


# Read image
img = cv2.imread('Images/sand_fft_lec_7.jfif')

# Convert to greyscale 
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Show original image
cv2.imshow('original image', img_grey)
cv2.waitKey(0)

# fft
img_fft = cv2.dft(np.float32(img_grey), flags=cv2.DFT_COMPLEX_OUTPUT)

# Shift the zero frequency component
img_fft_shift = np.fft.fftshift(img_fft)

# Calculate magnitude spectrum for visualization
img_fft_shift_abs = 10*np.log(cv2.magnitude(img_fft_shift[:,:,0], img_fft_shift[:,:,1]))

# This will truncate the decimal part and clip values outside [0, 255]
img_fft_shift_abs = img_fft_shift_abs.astype(np.uint8)

# Show fft magnitiude image
cv2.imshow('fft magnitude', img_fft_shift_abs)
cv2.waitKey(0)
