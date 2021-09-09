import cv2
import torch
import time
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable

from .hsm import HSMNet
from .utils_highres import CameraConfig, dry_run, round_size, set_disparity_range

drivingStereo_config = CameraConfig(0.546, 1000)

class HighResStereo():

	def __init__(self, model_path, config, camera_config = drivingStereo_config, use_gpu=True):

		self.use_gpu = use_gpu
		self.config = config
		self.camera_config = camera_config

		self.fps = 0
		self.timeLastPrediction = time.time()
		self.frameCounter = 0

		# Initialize model
		self.model = self.initialize_model(model_path, config, use_gpu)

	def __call__(self, left_img, right_img):

		return self.estimate_disparity(left_img, right_img)

	@staticmethod
	def initialize_model(model_path, config,  use_gpu):

		# construct net
		net = HSMNet(128,config.clean, level=config.qualityLevel)
		net = nn.DataParallel(net, device_ids=[0])

		if use_gpu:
			net.cuda()

		pretrained_dict = torch.load(model_path)
		pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
		net.load_state_dict(pretrained_dict['state_dict'],strict=False)

		# Dry run
		dry_run(net, use_gpu)

		# Set disparity range
		net = set_disparity_range(net, config)

		return net

	def estimate_disparity(self, left_img, right_img):
		self.timeStartPrediction = time.time()
	
		left_tensor = self.prepare_input(left_img)
		right_tensor = self.prepare_input(right_img)

		# Perform inference on the image
		self.disparity_map, entropy = self.inference(left_tensor, right_tensor)

		self.updateFps()

		return self.disparity_map

	def prepare_input(self, img):

		# Transform the image for inference
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		new_width = round_size(self.config.img_res_scale*img.shape[1])
		new_height = round_size(self.config.img_res_scale*img.shape[0])
		img = cv2.resize(img,(new_width, new_height)).astype(np.float32)

		# Scale input pixel values to -1 to 1
		mean=[0.485, 0.456, 0.406]
		std=[0.229, 0.224, 0.225]
		
		img = ((img/ 255.0 - mean) / std)
		img = img.transpose(2, 0, 1)
		img = img[np.newaxis,:,:,:] 

		if self.use_gpu:
			input_tensor = Variable(torch.FloatTensor(img).cuda())
		else:
			input_tensor = Variable(torch.FloatTensor(img))

		return input_tensor

	def inference(self, left_tensor, right_tensor):

		with torch.no_grad():
			disparity, entropy = self.model(left_tensor, right_tensor)

		disparity = torch.squeeze(disparity).data.cpu().numpy()
		entropy = torch.squeeze(entropy).data.cpu().numpy()

		invalid = np.logical_or(disparity == np.inf,disparity==0)
		disparity[invalid] = np.inf

		disparity = disparity.astype(np.uint8)

		return disparity, entropy

	def get_depth(self):
		return (self.camera_config.f*self.camera_config.baseline)/self.disparity_map

	def updateFps(self):

		ellapsedTime = time.time() - self.timeStartPrediction
		self.fps = int(1/ellapsedTime)
			