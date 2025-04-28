"""Generates an empty map."""

import cv2
import numpy as np

# Load the image
image = cv2.imread("stata_basement.png")

# List of obstacles
GRAY = (204, 204, 204)
WHITE = (255, 255, 255)
obstacles = [
    [(0, 0), (1729, 0), (1729, 1299), (0, 1299)],
]
fill = [WHITE]

# Draw lines and/or fill obstacles
for i, points in enumerate(obstacles):
    if fill[i]:
        cv2.fillPoly(image, [np.array(points)], color=fill[i])
    if fill[i] != WHITE:
        for j in range(len(points)):
            cv2.line(image, points[j], points[(j + 1) % len(points)], color=(0, 0, 0), thickness=2)

# Save the modified image
cv2.imwrite("empty_map.png", image)
