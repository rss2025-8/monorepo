import cv2
import numpy as np
import os
import shutil

def canny(image):
    """Apply Canny edge detection to the image"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    """Isolate the region of interest (ROI) using a triangular mask"""
    height = image.shape[0]
    width = image.shape[1]
    # Create a triangular region of interest
    # polygons = np.array([
    #     [(200, height), (width-100, height), (width//2, int(height*0.4))]
    # ])
    polygons = np.array([
        [(0, height), (width, height), (width, height//2), (0, height//2)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def make_coordinates(image, line_parameters):
    """Convert line parameters to x,y coordinates"""
    try:
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1*(3/5))

        # Check if slope is valid
        if abs(slope) < 0.001:  # Avoid near-horizontal lines
            return None

        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)

        # Ensure coordinates are within image bounds
        height, width = image.shape[:2]
        x1 = max(0, min(x1, width-1))
        x2 = max(0, min(x2, width-1))

        return np.array([x1, y1, x2, y2])
    except:
        return None

def average_slope_intercept(image, lines):
    """Calculate average slope and intercept for left and right lanes"""
    left_fit = []
    right_fit = []
    if lines is None:
        return None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # Check if line is not vertical (avoid division by zero)
        if abs(x2 - x1) > 0.1:  # Avoid very small differences
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            # Filter out unrealistic slopes
            if abs(slope) > 0.1 and abs(slope) < 10:  # Reasonable slope range
                # Separate left and right lanes based on slope
                if slope < 0:
                    left_fit.append((slope, intercept))
                else:
                    right_fit.append((slope, intercept))

    lines_to_draw = []
    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
        if left_line is not None and not np.isnan(left_line).any():
            lines_to_draw.append(left_line)

    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)
        if right_line is not None and not np.isnan(right_line).any():
            lines_to_draw.append(right_line)

    if len(lines_to_draw) > 0:
        return np.array(lines_to_draw)
    return None

# def display_lines(image, lines):
#     """Draw the detected lines on the image"""
#     line_image = np.zeros_like(image)
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line.reshape(4)
#             # Ensure all coordinates are valid integers
#             try:
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 # Check if coordinates are within image bounds
#                 height, width = image.shape[:2]
#                 if 0 <= x1 < width and 0 <= x2 < width and 0 <= y1 < height and 0 <= y2 < height:
#                     cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
#             except (TypeError, ValueError) as e:
#                 print(f"Invalid coordinates: {x1}, {y1}, {x2}, {y2}")
#                 continue
#     return line_image

def display_lines(image, lines):
    """Draw the detected lines on the image"""
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def prepare_output_directory(output_base_dir):
    """Create output directory, removing existing one if present"""
    if os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)

def get_all_image_paths(input_dir):
    """Get all image file paths from the input directory"""
    all_image_paths = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(os.path.join(root, file))
    return all_image_paths

def save_params(output_base_dir):
    """Save the current script to the output directory"""
    script_save_path = os.path.join(output_base_dir, "script.py")
    with open(__file__, 'r') as script, open(script_save_path, 'w') as save_script:
        save_script.write(script.read())

def create_save_directory(output_base_dir, image_path):
    """Create directory structure for saving results"""
    lane_name = os.path.basename(os.path.dirname(image_path))
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    save_dir = os.path.join(output_base_dir, lane_name, image_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir, image_name

def save_original_image(image_path, save_dir):
    """Copy original image to the save directory"""
    orig_image_path = os.path.join(save_dir, os.path.basename(image_path))
    shutil.copy(image_path, orig_image_path)

def save_image(image, save_dir, filename):
    """Save an image to the specified directory"""
    path = os.path.join(save_dir, filename)
    cv2.imwrite(path, image)

def process_image(image_path, params, output_base_dir, gallery_dir):
    """Process a single image for lane detection"""
    # Read the image
    image = cv2.imread(image_path)
    lane_image = np.copy(image)

    # Create save directory
    save_dir, image_name = create_save_directory(output_base_dir, image_path)

    # Save original image
    save_original_image(image_path, save_dir)

    # Step 1: Convert to grayscale and apply Canny edge detection
    canny_image = canny(lane_image)
    save_image(canny_image, save_dir, "canny_edge.jpg")

    # Step 2: Define region of interest
    cropped_image = region_of_interest(canny_image)
    save_image(cropped_image, save_dir, "region_of_interest.jpg")

    # Step 3: Apply Hough Transform
    lines = cv2.HoughLinesP(
        cropped_image,
        params['rho_resolution'],
        params['theta_resolution'],
        params['threshold'],
        # np.array([]),
        minLineLength=params['minLineLength'],
        maxLineGap=params['maxLineGap']
    )

    # Step 4: Average and extrapolate lanes
    averaged_lines = average_slope_intercept(lane_image, lines)

    # Step 5: Create line image
    if averaged_lines is not None:
        line_image = display_lines(lane_image, averaged_lines)
        save_image(line_image, save_dir, "detected_lines.jpg")

        # Step 6: Combine with original image
        combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
        save_image(combo_image, save_dir, "final_result.jpg")

        # Save gallery version
        gallery_path = os.path.join(gallery_dir, f"{image_name}_result.jpg")
        cv2.imwrite(gallery_path, combo_image)
    else:
        print(f"No lanes detected for {image_path}")

def process_and_save_images(params, input_dir, output_base_dir):
    """Process all images in the input directory"""
    prepare_output_directory(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)
    gallery_dir = os.path.join(output_base_dir, "gallery")
    os.makedirs(gallery_dir)
    save_params(output_base_dir)

    all_image_paths = get_all_image_paths(input_dir)
    for image_path in all_image_paths:
        process_image(image_path, params, output_base_dir, gallery_dir)

# Parameters from the tutorial
params = {
    "rho_resolution": 2,        # Rho resolution for Hough Transform
    "theta_resolution": np.pi/180,  # Theta resolution (1 degree)
    "threshold": 100,           # Threshold for line detection
    "minLineLength": 20,        # Minimum line length
    "maxLineGap": 50,           # Maximum gap between line segments
}

# Set paths
computer_vision_path = '/home/racecar/monorepo/src/final_challenge2025/race_to_the_moon/race_to_the_moon/computer_vision'
input_dir = os.path.join(computer_vision_path, "racetrack_images")
output_dir = os.path.join(computer_vision_path, "test_results_hough_lines_enhanced_7")

# Process images
process_and_save_images(params, input_dir, output_dir)

# Also provide a function to process video
def process_video(video_path, params, output_path):
    """Process a video for lane detection"""
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        canny_image = canny(frame)
        cropped_image = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(
            cropped_image,
            params['rho_resolution'],
            params['theta_resolution'],
            params['threshold'],
            # np.array([]),
            minLineLength=params['minLineLength'],
            maxLineGap=params['maxLineGap']
        )

        averaged_lines = average_slope_intercept(frame, lines)
        if averaged_lines is not None:
            line_image = display_lines(frame, averaged_lines)
            combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        else:
            combo_image = frame

        # Write frame
        out.write(combo_image)

        # Optional: Display the frame (press 'q' to quit)
        cv2.imshow('Lane Detection', combo_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage for video processing
# video_input = os.path.join(computer_vision_path, "test_video.mp4")
# video_output = os.path.join(output_dir, "processed_video.mp4")
# process_video(video_input, params, video_output)
