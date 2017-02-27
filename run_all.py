from moviepy.editor import VideoFileClip
import numpy as np
import cv2

from calibration import load_calibration_results

from lane import Lane
from lanes import Lanes

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import matplotlib.pyplot as plt

import glob
import matplotlib.image as mpimg

X_METER_PER_PIXEL = 3.7/700
Y_METER_PER_PIXEL = 30/720

class Line():
	def __init__(self):
		# was the line detected in the last iteration?
		self.detected = False  
		# x values of the last n fits of the line
		self.recent_xfitted = [] 
		#average x values of the fitted line over the last n iterations
		self.bestx = None     
		#polynomial coefficients averaged over the last n iterations
		self.best_fit = None  
		#polynomial coefficients for the most recent fit
		self.current_fit = [np.array([False])]  
		#radius of curvature of the line in some units
		self.radius_of_curvature = None 
		#distance in meters of vehicle center from the line
		self.line_base_pos = None 
		#difference in fit coefficients between last and new fits
		self.diffs = np.array([0,0,0], dtype='float') 
		#x values for detected line pixels
		self.allx = None  
		#y values for detected line pixels
		self.ally = None


class allStuff:
	def __init__(self):
		# for undistortion
		self.mtx, self.dist = load_calibration_results("./calibration_result.pkl")


		# for transformation
		self.src_points = np.float32([[585.714, 456.34],
										[699.041, 456.34],
										[1029.17, 667.617],
										[290.454, 667.617]])
		self.dst_points = np.float32([[300, 70],
										[1000, 70],
										[1000, 600], 
										[300, 600]])

		self.img_count = 1

		self.to_meters = np.array([[X_METER_PER_PIXEL, 0],
						[0, Y_METER_PER_PIXEL]])
	
	def p1(self, fit):
		"""first derivative"""
		return np.polyder(fit)


	def p2(self, fit):
		"""second derivative"""
		return np.polyder(fit, 2)

	# def curvature(self, y, fit):
	# 	"""returns the curvature of the of the lane in meters"""
	# 	return ((1 + (self.p1(fit)**2))**1.5) / np.absolute(self.p2(fit))
	
	def p(self, xs, ys):
		return np.poly1d(np.polyfit(ys, xs, 2))

	def in_meters(self, point):
		return np.dot(point, self.to_meters)

	def distance_from_center(self, center, leftx, rightx, lefty, righty):
		center = self.in_meters(center)
		center_x, center_y = center

		right_x = self.p(rightx, center_y)
		left_x = self.p(leftx, center_y)

		return ((right_x + left_x)/2 - center_x)

	# drawing lane functions
	def overlay_text(self, image, text, pos=(0, 0), color=(255, 255, 255)):
		image = Image.fromarray(image)
		draw = ImageDraw.Draw(image)
		font = ImageFont.truetype("./fonts/liberation-sans.ttf", 64)
		draw.text(pos, text, color, font=font)
		image = np.asarray(image)

		return image

	def overlay_lane(self, image, left_fit, right_fit):
		left_ys = np.linspace(0, 100, num=101) * 7.2
		left_xs = left_fit[0]*left_ys**2 + left_fit[1]*left_ys + left_fit[2]

		right_ys = np.linspace(0, 100, num=101) * 7.2
		right_xs = right_fit[0]*right_ys**2 + right_fit[1]*right_ys + right_fit[2]

		color_warp = np.zeros_like(image).astype(np.uint8)

		pts_left = np.array([np.transpose(np.vstack([left_xs, left_ys]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([right_xs, right_ys])))])
		pts = np.hstack((pts_left, pts_right))

		cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
		newwarp = cv2.warpPerspective(color_warp, self.inverse_transform_matrix(), (image.shape[1], image.shape[0]))
		newwarp = self.transform_from_top_down(color_warp, image)

		return cv2.addWeighted(image, 1, newwarp, 0.3, 0)

	def overlay_detected_lane_data(self, image, lanes):
		height, width, _ = image.shape

		image = self.overlay_lane(image, lanes.left.pixels.fit, lanes.right.pixels.fit)
		image = self.overlay_text(image, "Left curvature: {0:.2f}m".format(lanes.left.meters.curvature(height)), pos=(10, 10))
		image = self.overlay_text(image, "Right curvature: {0:.2f}m".format(lanes.right.meters.curvature(height)), pos=(10, 90))
		image = self.overlay_text(image, "Distance from center: {0:.2f}m".format(self.distance_from_center((width/2, height))), pos=(10, 170))

		return image

	def curvature(self, left_fit, right_fit, leftx, rightx, lefty, righty):
		ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
		y_eval = np.max(ploty)
		left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
		right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
		# print(left_curverad, right_curverad)

		# Define conversions in x and y from pixels space to meters
		ym_per_pix = 30/720 # meters per pixel in y dimension
		xm_per_pix = 3.7/700 # meters per pixel in x dimension

		# Fit new polynomials to x,y in world space
		left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
		right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
		# Calculate the new radii of curvature
		left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
		right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
		# Now our radius of curvature is in meters
		# print(left_curverad, 'm', right_curverad, 'm')
		# Example values: 632.1 m    626.2 m

		return left_curverad, right_curverad


	def draw_overlays(self, image, left_fit, right_fit, leftx, rightx, lefty, righty):
		height, width, _ = image.shape
		image = self.overlay_lane(image, left_fit, right_fit)


		

		left_curverad, right_curverad = self.curvature(left_fit, right_fit, leftx, rightx, lefty, righty)
		# print(left_curverad)
		# print(right_curverad)
		# distance = self.distance_from_center((width/2, height), leftx, rightx, lefty, righty)

		xm_per_pix = 3.7/700
		if len(lefty) > 1 and len(righty) > 1:
			lane_offset = (1280/2 - (left_fit[-1]+right_fit[-1])/2)*xm_per_pix
		else:
			lane_offset = 0
		# print(lane_offset)




		image = self.overlay_text(image, "Left curvature: {0:.2f}m".format(left_curverad), pos=(10, 10))
		image = self.overlay_text(image, "Right curvature: {0:.2f}m".format(right_curverad), pos=(10, 90))
		image = self.overlay_text(image, "Lane Offset: {0:.2f}m".format(lane_offset), pos=(10, 170))


		# if (self.img_count % 419 == 0):	
			# plt.imshow(image)
			# plt.savefig('./output_images/' + str(self.img_count) + '_output.png')
		self.img_count += 1
		
		return image

	# undistortion
	def undistort(self, img):
		# mtx, dist = load_calibration_results("./calibration_result.pkl")
		return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

	# transformation
	def transform_matrix(self):
		return cv2.getPerspectiveTransform(self.src_points, self.dst_points)

	def inverse_transform_matrix(self):
		return cv2.getPerspectiveTransform(self.dst_points, self.src_points)

	def transform_to_top_down(self, img):
		return cv2.warpPerspective(img, self.transform_matrix(), img.shape[1::-1])

	def transform_from_top_down(self, img, size_img):
		return cv2.warpPerspective(img, self.inverse_transform_matrix(), size_img.shape[1::-1])

	# thresholding
	def abs_sobel_thresholding(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
		thresh_min, thresh_max = thresh

		# Convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

		# Apply x or y gradient with the OpenCV Sobel() function
		# and take the absolute value
		if orient == 'x':
			abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
		elif orient == 'y':
			abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

		# Rescale back to 8 bit integer
		scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
		# Create a copy and apply the threshold
		binary_output = np.zeros_like(scaled_sobel)
		# Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too

		binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

		# Return the result
		return binary_output

	def saturation_thresholding(self, img, thresh=(0, 255)):
		# convert RGB image to HLS
		hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		# Use only the saturation channel
		S = hls[:,:,2]

		binary_output = np.zeros_like(S)
		binary_output[((S > thresh[0]) & (S < thresh[1]))] = 1
		return binary_output


	def red_thresholding(self, img, thresh=(0, 255)):
		# Use only the red channel
		R = img[:,:,0]

		# Create a binary mask where the red thresholds are met
		binary_output = np.zeros_like(R)
		binary_output[((R > thresh[0]) & (R < thresh[1]))] = 1
		return binary_output

	def thresholding_op(self, img):
		# Sobel kernel size
		ksize = 17 # choose a larger odd number to smooth gradient measurements

		gradx = self.abs_sobel_thresholding(img, orient="x", sobel_kernel=ksize, thresh=(15, 100))
		saturation = self.saturation_thresholding(img, thresh=(120, 255))
		reds = self.red_thresholding(img, thresh=(180, 255))

		combined = np.zeros_like(saturation)
		combined[(reds == 1) | (saturation == 1) | (gradx == 1)] = 1
		return combined



	# detect lane lines
	def detect_lane_lines(self, binary_warped):
		# print('detecting lane lines')
		# Assuming you have created a warped binary image called "binary_warped"
		# Take a histogram of the bottom half of the image
		histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
		# histogram = np.sum(binary_warped[binary_warped//2:,:], axis=0)
		
		# Create an output image to draw on and  visualize the result
		out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
		
		# Find the peak of the left and right halves of the histogram
		# These will be the starting point for the left and right lines
		midpoint = np.int(histogram.shape[0]/2)
		leftx_base = np.argmax(histogram[:midpoint])
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint

		# Choose the number of sliding windows
		nwindows = 9
		
		# Set height of windows
		window_height = np.int(binary_warped.shape[0]/nwindows)
		
		# Identify the x and y positions of all nonzero pixels in the image
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		
		# Current positions to be updated for each window
		leftx_current = leftx_base
		rightx_current = rightx_base

		# Set the width of the windows +/- margin
		margin = 100
		
		# Set minimum number of pixels found to recenter window
		minpix = 50
		
		# Create empty lists to receive left and right lane pixel indices
		left_lane_inds = []
		right_lane_inds = []

		# Step through the windows one by one
		for window in range(nwindows):
    		# Identify window boundaries in x and y (and right and left)
			win_y_low = binary_warped.shape[0] - (window+1)*window_height
			win_y_high = binary_warped.shape[0] - window*window_height
    		
			win_xleft_low = leftx_current - margin
			win_xleft_high = leftx_current + margin
    		
			win_xright_low = rightx_current - margin
			win_xright_high = rightx_current + margin
    
			# Draw the windows on the visualization image
			cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
			cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
    
			# Identify the nonzero pixels in x and y within the window
			good_left_inds = ((nonzeroy >= win_y_low) & 
								(nonzeroy < win_y_high) & 
								(nonzerox >= win_xleft_low) & 
								(nonzerox < win_xleft_high)).nonzero()[0]
			good_right_inds = ((nonzeroy >= win_y_low) & 
								(nonzeroy < win_y_high) & 
								(nonzerox >= win_xright_low) & 
								(nonzerox < win_xright_high)).nonzero()[0]

			# Append these indices to the lists
			left_lane_inds.append(good_left_inds)
			right_lane_inds.append(good_right_inds)
		
			# If you found > minpix pixels, recenter next window on their mean position
			if len(good_left_inds) > minpix:
				leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
			if len(good_right_inds) > minpix:        
				rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

		# Concatenate the arrays of indices
		left_lane_inds = np.concatenate(left_lane_inds)
		right_lane_inds = np.concatenate(right_lane_inds)

		# Extract left and right line pixel positions
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds] 
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds] 

		# Fit a second order polynomial to each
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)

		return leftx, lefty, rightx, righty, left_fit, right_fit, out_img

	def process_image(self, image):
		img_height, img_width, _ = image.shape

		# if (self.img_count % 419 == 0):
			# plt.imshow(image)
			# plt.savefig('./output_images/' + str(self.img_count) + '_input.png')

		bin_image = self.undistort(image)
		# if (self.img_count % 419 == 0):
			# plt.imshow(bin_image)
			# plt.savefig('./output_images/' + str(self.img_count) + '_undistort.png')
		
		bin_image = self.transform_to_top_down(bin_image)
		# if (self.img_count % 419 == 0):
			# plt.imshow(bin_image)
			# plt.savefig('./output_images/' + str(self.img_count) + '_topdown.png')

		bin_image = self.thresholding_op(bin_image)
		# if (self.img_count % 419 == 0):
			# plt.imshow(bin_image, 'gray')
			# plt.savefig('./output_images/' + str(self.img_count) + '_threshold.png')

		leftx, lefty, rightx, righty, left_fit, right_fit, out = self.detect_lane_lines(bin_image)

		# if (self.img_count % 419 == 0):
			# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 10))
			# f.tight_layout()

			# ax1.set_title("Sliding windows used in detection")
			# ax1.imshow(out, cmap="gray")

			# ax2.set_title("Fitted lines found in detection")
			# ax2.imshow(bin_image, cmap="gray");

			# # Left lane
			# left_p = np.poly1d(left_fit)
			# left_xp = np.linspace(0, 720, 100)
			# ax2.plot(left_p(left_xp), left_xp)

			# # Right lane
			# right_p = np.poly1d(right_fit)
			# right_xp = np.linspace(0, 720, 100)
			# ax2.plot(right_p(right_xp), right_xp)

			# f.savefig('./output_images/' + str(self.img_count) + '_slidewindow.png')
		

		return self.draw_overlays(
			image = image,
			left_fit = left_fit,
			right_fit = right_fit,
			leftx = leftx,
			rightx = rightx,
			lefty = lefty,
			righty = righty)




if __name__ == "__main__":
	np.seterr(all="ignore")

	stuff = allStuff()
	images_glob = glob.glob("camera_cal/calibration*.jpg")

	# image_shape = mpimg.imread(images_glob[0]).shape[1::-1]

	
	count = 1
	for filepath in images_glob:
		print(count)
		img = mpimg.imread(filepath)
		stuff.process_image(img)

		count += 1;


	# clip1 = VideoFileClip("./project_video.mp4")
	# white_clip = clip1.fl_image(stuff.process_image)
	# white_clip.write_videofile("./output_images/project_image.mp4", audio=False)