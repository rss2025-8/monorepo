import os

import cv2
import numpy as np
from final_challenge2025.shrinkray_heist.shrinkray_heist.traffic_light_detector import cd_color_segmentation, image_print

# test_folder = "./traffic_test_images"
# ground_truth = {
#     "1.png": True,
#     "2.png": True,
#     "3.png": True,
#     "4.png": False,
#     "5.png": False,
#     "6.png": True,
#     "7.png": False,
#     "8.png": False,
#     "9.png": False,
#     "10.png": True,
#     "11.png": False,
#     "12.png": False,
#     "13.png": True,
# }

# Iterate through images in test_folder
ground_truth = {}
for filename in os.listdir("test_images/light/green"):
    if filename.endswith(".png"):
        ground_truth[f"test_images/light/green/{filename}"] = "green"
for filename in os.listdir("test_images/light/red"):
    if filename.endswith(".png"):
        ground_truth[f"test_images/light/red/{filename}"] = "red"
for filename in os.listdir("test_images/light/yellow"):
    if filename.endswith(".png"):
        ground_truth[f"test_images/light/yellow/{filename}"] = "yellow"

correct = 0
total = 0

for filepath, true_label in ground_truth.items():
    img = cv2.imread(filepath)
    if img is not None:
        predicted = cd_color_segmentation(img)
        if true_label == "green" and predicted:
            correct += 1
        elif true_label == "yellow" and not predicted:
            correct += 1
        elif true_label == "red" and not predicted:
            correct += 1
        else:
            print(f"Incorrectly predicted {filepath} as {predicted}")
        total += 1

if total > 0:
    accuracy = correct / total
    print(f"\nOverall accuracy: {accuracy * 100:.2f}% ({correct}/{total} correct)")
else:
    print("No images were processed.")
