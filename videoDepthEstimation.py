import cv2
import pafy
import numpy as np

from highres_stereo import HighResStereo
from highres_stereo.utils_highres import Config, CameraConfig, draw_disparity, draw_depth, QualityLevel

# Initialize video
# cap = cv2.VideoCapture("video.mp4")

videoUrl = 'https://youtu.be/Yui48w71SG0'
videoPafy = pafy.new(videoUrl)
print(videoPafy.streams)
cap = cv2.VideoCapture(videoPafy.getbestvideo().url)

config = Config(clean=-1, qualityLevel = QualityLevel.High, max_disp=128, img_res_scale=1)
use_gpu = True
model_path = "models/final-768px.tar"

ret, frame = cap.read()

# Store baseline (m) and focal length (pixel)
input_width = frame.shape[1]/3
camera_config = CameraConfig(0.1, 0.5*input_width) # 90 deg. FOV
max_distance = 5

# Initialize model
highres_stereo_depth = HighResStereo(model_path, config, camera_config, use_gpu)

cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)	
while cap.isOpened():

	try:
		# Read frame from the video
		ret, frame = cap.read()
		if not ret:	
			break
	except:
		continue

	# Extract the left and right images
	left_img  = frame[:,:frame.shape[1]//3]
	right_img = frame[:,frame.shape[1]//3:frame.shape[1]*2//3]
	color_real_depth = frame[:,frame.shape[1]*2//3:]

	# Estimate the depth
	disparity_map = highres_stereo_depth(left_img, right_img)
	depth_map = highres_stereo_depth.get_depth()

	color_disparity = draw_disparity(disparity_map)
	color_depth = draw_depth(depth_map, max_distance)

	color_depth = cv2.resize(color_depth, (left_img.shape[1],left_img.shape[0]))
	combined_image = np.hstack((left_img,color_real_depth, color_depth))
	combined_image = cv2.putText(combined_image, f'{highres_stereo_depth.fps} fps', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2, cv2.LINE_AA)

	cv2.imshow("Estimated depth", combined_image)

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
