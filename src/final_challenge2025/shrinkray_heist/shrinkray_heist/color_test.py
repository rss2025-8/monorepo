import cv2
import numpy as np
import os

from color_segmentation import cd_color_segmentation, image_print

test_folder = "./traffic_test_images"

ground_truth = {
    '1.png': True,
    '2.png': True,
    '3.png': True,
    '4.png': False,
    '5.png': False,
    '6.png': True,
    '7.png': False,
    '8.png': False,
    '9.png': False,
    '10.png': True,
    '11.png': False,
    '12.png': False,
    '13.png': True,
}

correct = 0
total = 0

for filename, true_label in ground_truth.items():
    img_path = os.path.join(test_folder, filename)
    img = cv2.imread(img_path)

    if img is not None:
        predicted = cd_color_segmentation(img)
        is_correct = (predicted == true_label)
        correct += int(is_correct)
        total += 1

if total > 0:
    accuracy = correct / total
    print(f"\nOverall accuracy: {accuracy * 100:.2f}% ({correct}/{total} correct)")
else:
    print("No images were processed.")
