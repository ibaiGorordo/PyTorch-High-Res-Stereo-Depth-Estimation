import cv2
import numpy as np
from imread_from_url import imread_from_url

from highres_stereo import HighResStereo
from highres_stereo.utils_highres import Config, CameraConfig, draw_disparity, draw_depth, QualityLevel

if __name__ == '__main__':
    
    config = Config(clean=-1, qualityLevel = QualityLevel.High, max_disp=128, img_res_scale=1)

    use_gpu = True
    model_path = "models/final-768px.tar"

    # Load images
    left_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im2.png")
    right_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im6.png")

    # Initialize model
    highres_stereo_depth = HighResStereo(model_path, config, use_gpu=use_gpu)

    # Estimate the depth
    disparity_map = highres_stereo_depth(left_img, right_img)

    color_disparity = draw_disparity(disparity_map)
    color_disparity = cv2.resize(color_disparity, (left_img.shape[1],left_img.shape[0]))

    cobined_image = np.hstack((left_img, right_img, color_disparity))
    cobined_image = cv2.putText(cobined_image, f'{highres_stereo_depth.fps} fps', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2, cv2.LINE_AA)

    cv2.imwrite("out.jpg", cobined_image)

    cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)   
    cv2.imshow("Estimated disparity", cobined_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

