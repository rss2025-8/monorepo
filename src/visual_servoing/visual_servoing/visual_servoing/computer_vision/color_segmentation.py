import cv2
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

def cd_color_segmentation(img, template):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	########## YOUR CODE STARTS HERE ##########

	# img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21) # made the accuracy worse for some reason?

	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert bgr image to hsv

	# add a mask to allow for line following; make a mask that only allows for the bottom half of the image
	mask = np.zeros(hsv.shape[:2], dtype="uint8")
	# make a mask that passes 1/8 to 1/2 of the image
	
	mask[hsv.shape[0]//2:, :] = 255
	hsv = cv2.bitwise_and(hsv, hsv, mask=mask)

	# test 3
	# upper_color = np.array([27, 255, 255]) #upper-bound value
	# lower_color = np.array([5, 140, 153]) #lower-bound value

	# upper_color = np.array([65, 255, 255]) #upper-bound value
	# lower_color = np.array([5, 140, 150]) #lower-bound value

	# test 
	# upper_color = np.array([65, 255, 255]) #upper-bound value
	# lower_color = np.array([5, 160, 155]) #lower-bound value

	# test 5
	upper_color = np.array([10, 255, 255]) #upper-bound value
	lower_color = np.array([5, 200, 90]) #lower-bound value

	mask = cv2.inRange(hsv, lower_color, upper_color)

	# image_print(mask)

	# cv2.imshow("Mask", mask)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
  
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	bounding_box = ((0,0), (0, 0))

	if contours:
		largest_contour = max(contours, key=cv2.contourArea)
		x, y, w, h = cv2.boundingRect(largest_contour)
		bounding_box = ((x, y), (x + w, y + h))
	
	print(f"bounding box:	{bounding_box}")


	# cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

	# image_print(img)
	# ########### YOUR CODE ENDS HERE ###########

	# Return bounding box
	# cv2.imshow("Detected Cone", img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	
	return bounding_box

# best yet (w/o denoising - avg = 0.87)
# ('./test_images_cone/test1.jpg', 0.9754755060852547)
# ('./test_images_cone/test2.jpg', 0.9681285464283821)
# ('./test_images_cone/test3.jpg', 0.9686325187969925)
# ('./test_images_cone/test4.jpg', 0.9484536082474226)
# ('./test_images_cone/test5.jpg', 0.9300111982082867)
# ('./test_images_cone/test6.jpg', 0.6988961451771369)
# ('./test_images_cone/test7.jpg', 0.8993939393939394)
# ('./test_images_cone/test8.jpg', 0.9403546480386888)
# ('./test_images_cone/test9.jpg', 0.8214285714285714)
# ('./test_images_cone/test10.jpg', 0.9302536231884058)
# ('./test_images_cone/test11.jpg', 0.7462456680785522)
# ('./test_images_cone/test12.jpg', 0.8974358974358975)
# ('./test_images_cone/test13.jpg', 0.9843920145190562)
# ('./test_images_cone/test14.jpg', 0.684863523573201)
# ('./test_images_cone/test15.jpg', 0.6547619047619048)
# ('./test_images_cone/test16.jpg', 0.9526542324246772)
# ('./test_images_cone/test17.jpg', 0.7010869565217391)
# ('./test_images_cone/test18.jpg', 0.9035714285714286)
# ('./test_images_cone/test19.jpg', 0.9236842105263158)
# ('./test_images_cone/test20.jpg', 0.9226305609284333)

# with denoising(avg = 0.82)
# ('./test_images_cone/test1.jpg', 0.9754755060852547)
# ('./test_images_cone/test2.jpg', 0.9681285464283821)
# ('./test_images_cone/test3.jpg', 0.9686325187969925)
# ('./test_images_cone/test4.jpg', 0.9484536082474226)
# ('./test_images_cone/test5.jpg', 0.9300111982082867)
# ('./test_images_cone/test6.jpg', 0.6988961451771369)
# ('./test_images_cone/test7.jpg', 0.8993939393939394)
# ('./test_images_cone/test8.jpg', 0.9403546480386888)
# ('./test_images_cone/test9.jpg', 0.8214285714285714)
# ('./test_images_cone/test10.jpg', 0.9302536231884058)
# ('./test_images_cone/test11.jpg', 0.7462456680785522)
# ('./test_images_cone/test12.jpg', 0.8974358974358975)
# ('./test_images_cone/test13.jpg', 0.9843920145190562)
# ('./test_images_cone/test14.jpg', 0.684863523573201)
# ('./test_images_cone/test15.jpg', 0.6547619047619048)
# ('./test_images_cone/test16.jpg', 0.9526542324246772)
# ('./test_images_cone/test17.jpg', 0.7010869565217391)
# ('./test_images_cone/test18.jpg', 0.9035714285714286)
# ('./test_images_cone/test19.jpg', 0.9236842105263158)
# ('./test_images_cone/test20.jpg', 0.9226305609284333)

# img = cv2.imread('test_images_cone/cone.png')
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert bgr image to hsv
# h_min, s_min, v_min = np.min(hsv, axis=(0, 1))
# h_max, s_max, v_max = np.max(hsv, axis=(0, 1))
# print(f'Lower HSV: [{h_min}, {s_min}, {v_min}]')
# print(f'Upper HSV: [{h_max}, {s_max}, {v_max}]')
