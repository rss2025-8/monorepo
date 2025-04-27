from computer_vision.hough_line_utils import process_and_save_images
import os

computer_vision_path = '.'

process_and_save_images(computer_vision_path.racetrack_images, os.path.join(computer_vision_path, "test_results_hough_lines"))
