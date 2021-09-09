import numpy as np
import cv2
from dataclasses import dataclass
from enum import IntEnum
import torch
from torch.autograd import Variable
import math

from .hsm import disparityregression


class QualityLevel(IntEnum):
	Low = 3
	Medium = 2
	High = 1

class Config:

	def __init__(self, clean = -1, qualityLevel = QualityLevel.High, max_disp = 128, img_res_scale = 1):
		self.clean = clean # clean up output using entropy estimation
		self.qualityLevel = qualityLevel
		self.max_disp = max_disp# maximum disparity to search for
		self.img_res_scale = img_res_scale # Image resolution downscale ratio

@dataclass
class CameraConfig:
    baseline: float
    f: float

def draw_disparity(disparity_map):

	disparity_map = disparity_map.astype(np.uint8)
	norm_disparity_map = (255*((disparity_map-np.min(disparity_map))/(np.max(disparity_map) - np.min(disparity_map))))

	return cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map,1), cv2.COLORMAP_JET)

def draw_depth(depth_map, max_dist):
	
	norm_depth_map = 255*(1-depth_map/max_dist)
	norm_depth_map[norm_depth_map < 0] =0
	norm_depth_map[depth_map == 0] =0

	return cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map,1), cv2.COLORMAP_JET)

def dry_run(model, use_gpu):

	# dry run
	multip = 48
	imgL = np.zeros((1,3,24*multip,32*multip))
	imgR = np.zeros((1,3,24*multip,32*multip))

	if use_gpu:
		imgL = Variable(torch.FloatTensor(imgL).cuda())
		imgR = Variable(torch.FloatTensor(imgR).cuda())
	else:
		imgL = Variable(torch.FloatTensor(imgL))
		imgR = Variable(torch.FloatTensor(imgR))
		
	with torch.no_grad():
		model.eval()
		pred_disp,entropy = model(imgL,imgR)

def set_disparity_range(model, config):

	## change max disp
	tmpdisp = int(config.max_disp*config.img_res_scale//64*64)
	if (config.max_disp*config.img_res_scale/64*64) > tmpdisp:
		model.module.maxdisp = tmpdisp + 64
	else:
		model.module.maxdisp = tmpdisp
	if model.module.maxdisp ==64: model.module.maxdisp=128
	model.module.disp_reg8 =  disparityregression(model.module.maxdisp,16).cuda()
	model.module.disp_reg16 = disparityregression(model.module.maxdisp,16).cuda()
	model.module.disp_reg32 = disparityregression(model.module.maxdisp,32).cuda()
	model.module.disp_reg64 = disparityregression(model.module.maxdisp,64).cuda()

	return model


def round_size(size):
	return math.ceil(size / 64.0) * 64



