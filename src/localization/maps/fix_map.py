"""Adds Stata hallway obstacles to the map."""

import cv2
import numpy as np

# Load the image
image = cv2.imread("stata_basement.png")

# List of obstacles
GRAY = (204, 204, 204)
WHITE = (255, 255, 255)
obstacles = [
    [(1132, 276), (1130, 298), (1296, 298), (1296, 276)],
    [(1132, 276), (1296, 276)],
    [(882, 272), (882, 297), (781, 297), (781, 269)],
    [(1319, 308 - 18), (1319, 297 - 18), (1196 + 348, 300 - 18), (1196 + 348, 311 - 18)],
    [(1010 + 348, 375 - 7), (1209 + 348, 377 - 7), (1213 + 348, 361 - 7), (1007 + 348, 357 - 7)],
    [(1010 + 348, 375 - 7), (1209 + 348, 377 - 7)],
    [(1036, 275), (1036, 301), (960, 301), (936, 293), (936, 275)],
    [(1630, 984), (1630, 908), (1641, 908), (1641, 984)],
    [(1562, 984), (1562, 908), (1551, 908), (1551, 984)],
    [(1073, 275), (1073, 303), (1131, 303), (1131, 275)],
    [(1206, 276), (1206, 296), (1231, 296), (1231, 276)],
    [(1259, 276), (1259, 291), (1279, 291), (1279, 276)],
    [(1554, 628), (1576, 628), (1576, 497), (1554, 497)],
    [(1644, 472), (1644, 501), (1626, 501), (1626, 472)],
    [(1644, 472), (1644, 501)],
    [(0, 0), (1729, 0), (1729, 1299), (0, 1299)],
]
fill = [WHITE, None, GRAY, GRAY, WHITE, None, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, WHITE, None, None]

# Draw lines and/or fill obstacles
for i, points in enumerate(obstacles):
    if fill[i]:
        cv2.fillPoly(image, [np.array(points)], color=fill[i])
    if fill[i] != WHITE:
        for j in range(len(points)):
            cv2.line(image, points[j], points[(j + 1) % len(points)], color=(0, 0, 0), thickness=2)

# Save the modified image
cv2.imwrite("2025_stata_basement.png", image)
