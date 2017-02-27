import cv2
import matplotlib.image as mpimg
import glob
import numpy as np
import pickle

def save_calibration_results(filepath, mtx, dist):
	with open(filepath, "wb") as f:
		data = {}
		data["mtx"] = mtx
		data["dist"] = dist
		pickle.dump(data, f)

def load_calibration_results(filepath):
	with open(filepath, "rb") as f:
		data = pickle.load(f)
		mtx = data["mtx"]
		dist = data["dist"]
		return mtx, dist

def calibration():
	images_glob = glob.glob("camera_cal/calibration*.jpg")

	image_shape = mpimg.imread(images_glob[0]).shape[1::-1]

	cols = 9
	rows = 6
	chessboard_shape = (cols, rows)

	objp_arr = []
	imgp_arr = []

	count = 1;
	for filepath in images_glob:
		print(count)
		img = mpimg.imread(filepath)
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, chessboard_shape, None)

		objp = np.zeros((cols*rows, 3), np.float32)
		objp[:,:2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

		if ret:
			objp_arr.append(objp)
			imgp_arr.append(corners)

		count += 1;

	_, mtx, dist, _, _ = cv2.calibrateCamera(objp_arr, imgp_arr, image_shape, None, None)

	save_calibration_results("./calibration_result.pkl", mtx, dist)

if __name__ == "__main__":
	calibration()