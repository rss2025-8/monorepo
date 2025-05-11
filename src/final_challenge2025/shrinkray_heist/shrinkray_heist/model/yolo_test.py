import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from detector import Detector
from PIL import Image


def image_print(img):
    """
    Helper function to print out images, for debugging. Pass them in as a list.
    Press any key to continue.
    """
    # Convert PIL image to RGB
    img = img.convert("RGB")
    plt.imshow(img)
    plt.show()


# Iterate through images in test_folder
ground_truth = {}
for filename in os.listdir("../test_images/banana/banana"):
    if filename.endswith(".png"):
        ground_truth[f"../test_images/banana/banana/{filename}"] = "banana"
for filename in os.listdir("../test_images/banana/none"):
    if filename.endswith(".png"):
        ground_truth[f"../test_images/banana/none/{filename}"] = "none"

correct = 0
correct_banana = 0
total = 0

detector = Detector()
detector.set_threshold(0.25)

for filepath, true_label in ground_truth.items():
    img = cv2.imread(filepath)
    # Convert to RGB
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if img is not None:
        results = detector.predict(img, silent=False)

        predictions = results["predictions"]
        original_image = results["original_image"]

        banana_bounding_boxes = [point for point, label in predictions if label in ("banana", "frisbee")]

        if banana_bounding_boxes:
            xmin, ymin, xmax, ymax = banana_bounding_boxes[0]
        else:
            pass

        boxed_img = detector.draw_box(original_image, predictions, draw_all=True)
        # image_print(pil_img)
        print("Predicted!")
        image_print(boxed_img)
        if true_label == "banana" and banana_bounding_boxes:
            correct += 1
            correct_banana += 1
        elif true_label == "none" and not banana_bounding_boxes:
            correct += 1
        else:
            print(f"Incorrectly predicted {filepath}, ground truth {true_label}")
        total += 1

if total > 0:
    accuracy = correct / total
    print(f"\nOverall accuracy: {accuracy * 100:.2f}% ({correct}/{total} correct)")
    print(f"# banana correct: {correct_banana}, # none correct: {correct - correct_banana}")
else:
    print("No images were processed.")
