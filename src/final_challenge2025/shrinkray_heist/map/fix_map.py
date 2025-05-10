"""Adds Stata hallway obstacles to the map."""

import cv2
import numpy as np

# Load the image
image = cv2.imread("old_stata_basement_obs.png")

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
    # [(1563, 982), (1563, 996), (1530, 996), (1530, 982)],
    [(1317, 260), (1542, 260), (1542, 386), (1317, 386)],  # Prevents planning the long way around the map
    [(804, 539), (798 - 2, 578 - 2), (740 - 2, 641 - 2), (726 - 2, 634 - 2), (757, 538)],  # Traffic light funnel 1
    [(863, 555), (851 + 2, 545 + 2), (755 + 2, 652 + 2), (765 + 2, 660 + 2)],  # Traffic light funnel 2
    [(667, 806), (647, 818), (652, 827), (670, 815)],  # Obstacle 1
    [(839, 840), (834, 845), (843, 854), (848, 848)],  # Obstacle 2
    [(1007, 1019), (1021, 1019), (1021, 1039), (1007, 1039)],  # Obstacle 3
    # [(942, 984), (932, 984), (929, 822), (941, 818)],  # 3rd banana obstacle
    [(933, 869), (923, 986), (945, 986), (942, 870)],  # 3rd banana obstacle (guides toward banana)
    # [(890, 959), (923, 959), (923, 975), (890, 975)],  # Banana obstacle
    # [(791, 671), (683, 597), (570, 721), (711, 776)],  # Prevents planning through the main hallway
]
fill = [
    WHITE,
    None,
    GRAY,
    GRAY,
    WHITE,
    None,
    GRAY,
    GRAY,
    GRAY,
    GRAY,
    GRAY,
    GRAY,
    GRAY,
    WHITE,
    None,
    None,
    GRAY,
    GRAY,
    GRAY,
    GRAY,
    GRAY,
    GRAY,
    GRAY,
]

# Draw lines and/or fill obstacles
for i, points in enumerate(obstacles):
    if fill[i]:
        cv2.fillPoly(image, [np.array(points)], color=fill[i])
    if fill[i] != WHITE:
        for j in range(len(points)):
            cv2.line(image, points[j], points[(j + 1) % len(points)], color=(0, 0, 0), thickness=2)

# Save the modified image
cv2.imwrite("stata_main_plan.png", image)
