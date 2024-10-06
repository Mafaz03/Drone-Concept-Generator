import numpy as np
import cv2

def resize_and_pad_image(img_array, target_size):
    # Get the original dimensions
    h, w = img_array.shape[:2]

    # Calculate aspect ratios
    aspect_ratio = w / h
    target_aspect_ratio = target_size[0] / target_size[1]

    if aspect_ratio > target_aspect_ratio:
        new_w = target_size[0]
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = target_size[1]
        new_w = int(new_h * aspect_ratio)

    # Resize the image while maintaining the aspect ratio
    resized_img = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a new image with the target size and white padding
    new_img = np.full((target_size[1], target_size[0], 3), 255, dtype=np.uint8)

    # Calculate position for pasting the resized image
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2

    # Paste the resized image onto the new image
    new_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

    return new_img
