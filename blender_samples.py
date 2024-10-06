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

# Set the resolution for a 4K blank canvas
blank = np.zeros((2160, 3840, 3), dtype=np.uint8)

# User input: specify the distance between spheres
distance_between_spheres = 15  # Example: maintain a distance of 15 pixels between spheres

# Adjust contours and sample points
scale_factor = 3840 / img.shape[1]  # Assuming the original image width
scaled_contours = [con * scale_factor for con in no_dupes]

# Draw sampled points on the 4K mask (without drawing contour lines)
for idx, con in enumerate(scaled_contours):
    con = con.astype(np.int32)  # Ensure the contour has the correct type
    if len(con) == 0:
        continue  # Skip empty contours
    
    # Sample points with the specified distance between them
    sampled_points = sample_contour_points_by_distance(con, distance_between_spheres)
    
    if len(sampled_points) == 0:
        continue  # Skip if no points were sampled

    # Draw the sampled points on the blank canvas
    for point in sampled_points:
        cv2.circle(blank, tuple(point.astype(int)), 2, (255, 255, 255), -1)  # Blue point

# Convert the BGR image (OpenCV format) to RGB for saving or displaying in matplotlib
blank_rgb = cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)

# Save the final output image (as a PNG, for example)
output_path = "output_image.png"
cv2.imwrite(output_path, cv2.cvtColor(blank_rgb, cv2.COLOR_RGB2BGR))

# Optionally display the image using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(blank_rgb)
plt.axis('off')  # Hide axes
plt.show()

print(f"Image saved as {output_path}")

