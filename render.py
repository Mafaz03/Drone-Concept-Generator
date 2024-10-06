import cv2
import numpy as np
import matplotlib.pyplot as plt

def sample_contour_points_by_distance(contour, distance_between_points):
    # Ensure contour is of correct data type (int32 or float32)
    contour = contour.astype(np.float32)

    # Check if the contour is valid (non-empty)
    if len(contour) == 0:
        return np.array([])  # Return an empty array if contour is empty

    # Get the total length of the contour
    contour_length = cv2.arcLength(contour, True)

    # Initialize variables for sampling
    sampled_points = []
    distance_accumulator = 0
    current_point = contour[0][0]

    sampled_points.append(current_point)  # Add the first point

    # Iterate through the contour and sample points
    for i in range(1, len(contour)):
        pt1 = current_point
        pt2 = contour[i][0]

        # Calculate the distance between consecutive points
        segment_length = np.linalg.norm(pt2 - pt1)

        while distance_accumulator + segment_length >= distance_between_points:
            # Calculate interpolation ratio
            ratio = (distance_between_points - distance_accumulator) / segment_length
            interpolated_point = pt1 + ratio * (pt2 - pt1)
            sampled_points.append(interpolated_point)
            current_point = interpolated_point
            distance_accumulator = 0
            segment_length -= distance_between_points - distance_accumulator
        distance_accumulator += segment_length
        current_point = pt2

    return np.array(sampled_points)

def draw_glowing_circle(image, center, color, radius=3, intensity=255):
    """Draws a glowing circle by creating a bloom effect with increasing blur and decreasing opacity."""
    glow_layers = 3  # Set to 1 for a single glow layer

    for i in range(glow_layers):
        # For the first layer, use full intensity without decay
        if i == 0:
            # Create the glow layer with full intensity
            glow_layer = np.zeros_like(image)
            b, g, r = map(int, color)  # Map color values to integers
            cv2.circle(glow_layer, center, radius, (b, g, r), -1)  # Draw the circle without blur
            
            # Add the first layer directly to the original image
            image = cv2.addWeighted(image, 1.0, glow_layer, 1.0, 0)  # Full intensity for the first layer
        else:
            # For subsequent layers, decrease intensity for a softer glow
            alpha = max(int(intensity / (2 ** i)), 5)  # Use a decay factor of 2 for softer fading
            blur_radius = (i + 1) * 6  # Increase blur for each layer
            
            # Ensure blur_radius is an odd number
            if blur_radius % 2 == 0:
                blur_radius += 5

            # Create a separate layer to apply the blur and glow
            glow_layer = np.zeros_like(image)
            b, g, r = map(int, color)  # Map color values to integers
            cv2.circle(glow_layer, center, radius, (b, g, r), -1)  # Draw the circle
            
            # Apply Gaussian blur for this layer
            glow_layer = cv2.GaussianBlur(glow_layer, (blur_radius, blur_radius), 0)

            # Add the blurred layer back onto the original image with reduced intensity
            image = cv2.addWeighted(image, 1.0, glow_layer, alpha / 255.0, 0)  # Apply alpha for subsequent layers

    return image

img_height, img_width = img.shape[:2]

# Set the resolution for a blank canvas, ensuring it matches the image dimensions
blank = np.zeros((img_height, img_width, 3), dtype=np.uint8)

# Use the original image dimensions for scaling contours
scale_factor_width = img_width / img.shape[1]  # Scale factor based on width
scale_factor_height = img_height / img.shape[0]  # Scale factor based on height

# Adjust contours and sample points
scaled_contours = [con * (scale_factor_width, scale_factor_height) for con in no_dupes]

distance_between_spheres = 6

# Draw sampled points with glow effect on the blank canvas (without drawing contour lines)
for idx, con in tqdm(enumerate(scaled_contours), total=len(scaled_contours)):
    con = con.astype(np.int32)  # Ensure the contour has the correct type
    if len(con) == 0:
        continue  # Skip empty contours
    
    # Sample points with the specified distance between them
    sampled_points = sample_contour_points_by_distance(con, distance_between_spheres)
    
    if len(sampled_points) == 0:
        continue  # Skip if no points were sampled

    # Draw the glowing points on the blank canvas
    for point in sampled_points:
        # Ensure the point is within the bounds of the image
        x, y = tuple(point.astype(int))
        if x >= img_width or y >= img_height or x < 0 or y < 0:
            continue
        
        # Sample the color from the original image
        color = img[y, x]  # BGR format
        
        # Draw the glowing circle with the sampled color
        blank = draw_glowing_circle(blank, (x, y), color)

# Convert the BGR image (OpenCV format) to RGB for saving or displaying in matplotlib
blank_rgb = cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)

# Save the final output image (as a PNG, for example)
output_path = "output_image_glow_sampled_colors.png"
cv2.imwrite(output_path, cv2.cvtColor(blank_rgb, cv2.COLOR_RGB2BGR))

# Optionally display the image using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(blank_rgb)
plt.axis('off')  # Hide axes
plt.show()

print(f"Image saved as {output_path}")