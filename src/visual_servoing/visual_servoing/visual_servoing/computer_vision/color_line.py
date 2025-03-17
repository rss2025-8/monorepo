import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def cd_color_segmentation(image, template):
  hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  lower_orange = np.array([5, 150, 150])   # adjust for lighting conditions
  upper_orange = np.array([15, 255, 255])

  height, width, _ = image.shape
  crop_height = 50  # adjust this depending on your camera setup
  roi = hsv_image[height - crop_height : height, 0:width]  # bottom strip of the image

  mask = cv2.inRange(roi, lower_orange, upper_orange)

  kernel = np.ones((5, 5), np.uint8)
  mask = cv2.erode(mask, kernel, iterations=2)
  mask = cv2.dilate(mask, kernel, iterations=2)

  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  if contours:
      largest_contour = max(contours, key=cv2.contourArea)

      M = cv2.moments(largest_contour)
      if M["m00"] != 0:
          cx = int(M["m10"] / M["m00"])  # Centroid X position
          cy = int(M["m01"] / M["m00"])  # Centroid Y position
      else:
          cx, cy = width // 2, crop_height // 2  # Default to center if no line detected

      cv2.circle(image, (cx, height - crop_height + cy), 5, (0, 255, 0), -1)

      image_print(image)

      # Determine steering direction
      error = cx - (width // 2)  # How far off-center the line is
      if error < -20:
          direction = "Turn Left"
      elif error > 20:
          direction = "Turn Right"
      else:
          direction = "Go Straight"

      print(f"Centroid at: ({cx}, {cy}), Steering: {direction}")

if __name__ == '__main__':
    # Keep track of scores
    scores = {}
    # Open test images csv
    with open(csv_file_path) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        # Iterate through all test images
        for row in csvReader:
            # Find image path and ground truth bbox
            img_path = row[0]
            bbox_true = ast.literal_eval(row[1])
            if not swap:
                img = cv2.imread(img_path)
                template = cv2.imread(template_file_path, 0)
            else:
                template = cv2.imread(img_path, 0)
                img = cv2.imread(template_file_path)
            # Detection bbox
            bbox_est = detection_func(img, template)

            # cv2.rectangle(img, bbox_est[0], bbox_est[1], color=(0, 255, 0), thickness=2)
            # cv2.rectangle(img, bbox_true[0], bbox_true[1], color=(255, 0, 0), thickness=2)
            image_print(img)
            
            score = iou_score(bbox_est, bbox_true)
            
            # Add score to dict
            scores[img_path] = score

    # Return scores
    return scores