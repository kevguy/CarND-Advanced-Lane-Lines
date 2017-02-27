##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"
[image7]: ./output_images/chessboard/1_input.png "Chessboard 1 Original"
[image8]: ./output_images/chessboard/1_undistort.png "Chessboard 1 Undistorted"
[image9]: ./output_images/chessboard/3_input.png "Chessboard 3 Original"
[image10]: ./output_images/chessboard/3_undistort.png "Chessboard 3 Undistorted"
[image11]: ./output_images/chessboard/19_input.png "Chessboard 19 Original"
[image12]: ./output_images/chessboard/19_undistort.png "Chessboard 19 Undistorted"
[image13]: ./output_images/output/419_input.png "Input 419"
[image14]: ./output_images/output/419_topdown.png "Input 419 (Topdown)"
[image15]: ./output_images/output/838_input.png "Input 838"
[image16]: ./output_images/output/838_topdown.png "Input 838 (Topdown)"
[image17]: ./output_images/output/1257_input.png "Input 1257"
[image18]: ./output_images/output/1257_topdown.png "Input 1257 (Topdown)"
[image19]: ./output_images/output/419_input.png "Input 419"
[image20]: ./output_images/output/419_threshold.png "Input 419 (Thresholding)"
[image21]: ./output_images/output/838_input.png "Input 838"
[image22]: ./output_images/output/838_threshold.png "Input 838 (Thresholding)"
[image23]: ./output_images/output/1257_input.png "Input 1257"
[image24]: ./output_images/output/1257_threshold.png "Input 1257 (Thresholding)"
[image25]: ./output_images/output/419_slidewindow.png "Input 419 (Slidewindow)"
[image26]: ./output_images/output/838_slidewindow.png "Input 1257 (Slidewindow)"
[image27]: ./output_images/output/1257_slidewindow.png "Input 1257 (Slidewindow)"
[image28]: ./output_images/output/419_output.png "Input 419 (Output)"
[image29]: ./output_images/output/838_output.png "Input 1257 (Output)"
[image30]: ./output_images/output/1257_output.png "Input 1257 (Output)"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


###Camera Calibration
####1. Have the camera matrix and distortion coefficients been computed correctly and checked on one of the calibration images as a test?

The code for this step is contained in the file `calibration.py`. I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion
coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to the test
image using the `cv2.undistort()` function and obtained this result:

Chessboard 1 Original
![alt text][image7]
Chessboard 1 Undistorted
![alt text][image8]
Chessboard 3 Original
![alt text][image9]
Chessboard 3 Undistorted
![alt text][image10]
Chessboard 19 Original
![alt text][image11]
Chessboard 19 Undistorted
![alt text][image12]


###Transforming an image

The code for my perspective transform is included in a function called transform_to_top_down(). Each image is transformed to a top-down view in order to clearly see the lane. The warper() function takes as inputs an image ( img ), as well as source ( src ) and destination ( dst ) points. I chose to hardcode the source and destination points in the following manner:


| Source               | Destination    | 
|:--------------------:|:--------------:| 
| 585.714, 456.34      | 300, 70        | 
| 699.041, 456.34      | 1000, 70       |
| 1029.17, 667.617     | 1000, 600      |
| 290.454, 667.617     | 300, 600        |



Here is an example of a test image transformed to a top-down view.

Input 419
![alt text][image13]
Input 419 (Topdown)
![alt text][image14]
Input 838
![alt text][image15]
Input 838 (Topdown)
![alt text][image16]
Input 1257
![alt text][image17]
Input 1257 (Topdown)
![alt text][image18]


###Thresholding an image
After an image is undistorted and transformed to a top-down view, it goes through the process of thresholding. 

After some trial and error, the following thresholding techniques worked best:

####HLS - Saturation channel thresholding
After the converting the RGB image to the HLS colorspace, I inly kept the saturation channel, which is comparably for detecting yellow lines.

####RGB - Red channel thresholding
The red channel is for detecting the white lines.

####Sobel gradient thesholding in X direction
Using sobel gradient thresholding I was able to detect changes in horizontal direction.

####The whole approach
It's just an OR combination of all the three techniques.



Here are some examples:
Input 419
![alt text][image19]
Input 419 (Thresholding)
![alt text][image20]
Input 838
![alt text][image21]
Input 838 (Thresholding)
![alt text][image22]
Input 1257
![alt text][image23]
Input 1257 (Thresholding)
![alt text][image24]



###Detecting lane lines
Lane line detection is done using the sliding windows approach. First we take all the line points detected for the bottom half of the image and detect theleft and right lane lines. We define a window and find all the nonzero values inside, store them as lane points, then slide up the window and repeat. Fially, we fit a parabola for the points detected, which we can use for lane line calculation for any point later.

Input 419 (Slidewindow)
![alt text][image25]
Input 838 (Slidewindow)
![alt text][image26]
Input 1257 (Slidewindow)
![alt text][image27]


###Lane Curvature
Lane curvature is determined using the following function:


```
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
```

###Distance from center
Assuming the camera is mounted in the center of the car, the distance to the center of the lane can be calculated by finding the difference from `the center of the lanes` to `the center of the image`.



```
xm_per_pix = 3.7/700
if len(lefty) > 1 and len(righty) > 1:
	lane_offset = (1280/2 - (left_fit[-1]+right_fit[-1])/2)*xm_per_pix
else:
	lane_offset = 0
```

Input 419 (Output)
![alt text][image28]
Input 838 (Output)
![alt text][image29]
Input 1257 (Output)
![alt text][image30]


###Pipeline (video)

A pipeline can be constructed using all the above operations, which can be found in the function `process_image()`:



```
def process_image(self, image):

	img_height, img_width, _ = image.shape
		
	# undistortion
	bin_image = self.undistort(image)
	
	# transformatoin
	bin_image = self.transform_to_top_down(bin_image)
	
	# thresholding
	bin_image = self.thresholding_op(bin_image)
	

	# detect lane lines
	leftx, lefty, rightx, righty, left_fit, right_fit, _ = self.detect_lane_lines(bin_image)

	# draw the results onto the image
	return self.draw_overlays(
		image = image,
		left_fit = left_fit,
		right_fit = right_fit,
		leftx = leftx,
		rightx = rightx,
		lefty = lefty,
		righty = righty)
```
Here's a [link to my video result](https://youtu.be/RxcDBK14jNc)


### Conclusion
The whole approach works on the project video, but not the challenge video. I guess there's more to the thresholding approach I adopt that needs to be perfected since the current thresholding approach may introduce 'outliers' that we don't want, which may need further thresholidng to eliminate.

