import numpy as np
import cv2


def change_resolution(img, scale_factor):
    """Resize image by scale_factor."""
    height, width = img.shape[:2]
    new_size = (int(width * scale_factor), int(height * scale_factor))
    resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return resized_img

def quantize_image(img, levels):
    """Quantize image to the given number of levels."""
    img_float = img.astype(np.float32)
    max_val = 255
    step = max_val // (levels - 1)
    quantized = np.round(img_float / step) * step
    return np.clip(quantized, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    # Load image
    # img = cv2.imread("Images/res_quant_lec_3.jpg") # Change to your image path
    img = cv2.imread("Images/res_quant_lec_3.jfif") # Change to your image path 


    # Change resolution
    scale_factor = 0.5  # e.g., reduce to 50%
    img_resized = change_resolution(img, scale_factor)

    # Change quantization
    quant_levels = 4 # e.g., 8 gray levels per channel
    # img_quantized = quantize_image(img_resized, quant_levels)
    img_quantized = quantize_image(img, quant_levels)


    # Save or display results
    cv2.imwrite("output_resized.jpg", img_resized)
    cv2.imwrite("output_quantized.jpg", img_quantized)
    cv2.imshow("Resized Image", img_resized)
    cv2.waitKey(0)
    cv2.imshow("Quantized Image", img_quantized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()