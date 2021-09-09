import cv2
import pafy
import numpy as np
import glob

from highres_stereo import HighResStereo
from highres_stereo.utils_highres import Config, CameraConfig, draw_disparity, draw_depth, QualityLevel

# out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (881*3,400))

# Get image list
left_images = glob.glob('DrivingStereo images/left/*.jpg')
left_images.sort()
right_images = glob.glob('DrivingStereo images/right/*.jpg')
right_images.sort()
depth_images = glob.glob('DrivingStereo images/depth/*.png')
depth_images.sort()

config = Config(clean=-1, qualityLevel = QualityLevel.High, max_disp=128, img_res_scale=1)
use_gpu = True
model_path = "models/final-768px.tar"

left_img = cv2.imread(left_images[0])
input_width = left_img.shape[1]
camera_config = CameraConfig(0.546, 2000/1920*input_width) # rough estimate from the original calibration
max_distance = 30

# Initialize model
highres_stereo_depth = HighResStereo(model_path, config, camera_config, use_gpu)

cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)	
for left_path, right_path, depth_path in zip(left_images[:], right_images[:], depth_images[:]):

	# Read frame from the video
	left_img = cv2.imread(left_path)
	right_img = cv2.imread(right_path)
	depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)/256

	# Estimate the depth
	disparity_map = highres_stereo_depth(left_img, right_img)
	depth_map = highres_stereo_depth.get_depth()

	color_disparity = draw_disparity(disparity_map)
	color_depth = draw_depth(depth_map, max_distance)
	color_real_depth = draw_depth(depth_img, max_distance)

	color_depth = cv2.resize(color_depth, (left_img.shape[1],left_img.shape[0]))
	combined_image = np.hstack((left_img,color_real_depth, color_depth))
	combined_image = cv2.putText(combined_image, f'{highres_stereo_depth.fps} fps', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2, cv2.LINE_AA)

	# out.write(cobined_image)
	cv2.imshow("Estimated depth", cobined_image)

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

# out.release()
cv2.destroyAllWindows()
