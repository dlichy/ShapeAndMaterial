from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os
import glob
import torch
import torchvision
import sys
#sys.path.append("..")
from utils.exr_read_write_funs import read_exr
import scipy.ndimage
import random
import json
import cv2
#random.seed(0)



class SyntheticDataset(Dataset):

	#mode = 'select' or 'random' 
	#if mode is 'select' than argument selected_lights should be a list of light directions to load
	#if mode is 'random' than a random sample of light directions will be loaded. skip_num_prob should be a list of probabilities of length 6 s.t. skip_num_prob[i] is the probability that i images will be skipped. Note the central image is never skipped
	def __init__(self, dataset_root, train, domains_to_load = ['albedo','rough','normal','depth','mask','env_image'], colocated_as_2=False, transform=None, mode='select', selected_lights=[x for x in range(0,6)], skip_num_prob=[0.1,0.1,0.1,0.2,0.2,0.3] ):
		
		self.mode = mode
		self.selected_lights = selected_lights
		self.to_skip = [x for x in range(0,6) if x not in selected_lights]
		self.skip_num_prob = skip_num_prob
		self.colocated_as_2 = colocated_as_2
		self.domains_to_load = domains_to_load
		self.transform = transform
		self.set_scene_list(dataset_root, train)
		
	def set_scene_list(self,dataset_root,train):
		if train:
			p = os.path.join(dataset_root,'dataset/train/scene_*')
		else:
			p = os.path.join(dataset_root,'dataset/test/scene_*')
		
		scene_list = glob.glob(p)
		print('found {} scenes in scene list'.format(len(scene_list)))
		self.scene_list = scene_list
	
	def __len__(self):
		return len(self.scene_list)

	#a scene dict has fields: int scene_num, int light_num, list of ints view_nums
	#first view_num is the reference scene. If scale coord by median depth in the mask
	def __getitem__(self, idx):
		scene_path = self.scene_list[idx]
		xml_path = scene_path.replace('dataset','xmls')
		
		cam_to_world_path =  os.path.join(xml_path, 'cam_to_world.txt')
		cam_to_world  = np.loadtxt(cam_to_world_path)
		world_to_cam = cam_to_world[0:3,0:3].transpose().astype('single')
		
		sample = {}
		sample['scene_path'] = scene_path
		sample['name'] = os.path.basename(scene_path)

		if 'albedo' in self.domains_to_load:
			albedo = read_exr( os.path.join(scene_path, 'albedo.exr'))
			sample['albedo'] = albedo
		if 'depth' in self.domains_to_load:
			depth = read_exr( os.path.join(scene_path, 'depth.exr'), channel_names = ['Y'])
			sample['depth'] = depth
		if 'mask' in self.domains_to_load:
			mask = read_exr( os.path.join(scene_path,'mask.exr'), channel_names = ['Y'])
			sample['mask'] = mask
		if 'rough' in self.domains_to_load:
			roughness = read_exr( os.path.join(scene_path, 'roughness.exr'), channel_names = ['Y'])
			sample['rough'] = roughness
		if 'normal' in self.domains_to_load:
			normal = read_exr( os.path.join(scene_path,'normal.exr'))
			s = normal.shape
			normal = np.matmul(normal.reshape(-1,3),world_to_cam.transpose()).reshape(s)
			sample['normal'] = normal
		if 'env_image' in self.domains_to_load:
			env_image = read_exr( os.path.join(scene_path, 'image_env.exr'))
			sample['env_image'] = env_image
		
		with open(os.path.join(xml_path,'lights.json')) as f:
			all_lights = json.load(f)
		
		images = []
		light_dirs = []
		light_irr = []

		if self.mode == 'random':
			num_to_skip = random.choices([0,1,2,3,4,5],weights=self.skip_num_prob)[0]
			lights_to_skip = random.sample([0,1,3,4,5],num_to_skip)
		elif self.mode == 'select':
			lights_to_skip = self.to_skip
		else:
			print('mode not recognized')
			
		for light_num in range(0,6):
			#if skipping add phony light directions
			if light_num in lights_to_skip:
				images.append(np.zeros((256,256,3),dtype=np.float32))
				light_dir = np.array(all_lights[light_num]['direction']).reshape(3,1)
				light_dir = np.matmul(world_to_cam,light_dir)
				light_dirs.append(light_dir.squeeze().astype('single'))
				light_irr.append(all_lights[light_num]['irradiance'])
				continue
			
			if self.colocated_as_2 and light_num == 2:
				light_num = -1
				
			image = read_exr( os.path.join(scene_path, 'image_{:0>2}.exr'.format(light_num+1)))
			images.append(image)
			
			light_dir = np.array(all_lights[light_num]['direction']).reshape(3,1)
			light_dir = np.matmul(world_to_cam,light_dir)
			light_dirs.append(light_dir.squeeze().astype('single'))
			light_irr.append(all_lights[light_num]['irradiance'])
			
			
		sample['light_dirs'] = light_dirs
		sample['light_irr'] = light_irr
		sample['images'] = images
			
		if self.transform:
			self.transform(sample)
		
		return sample
	
if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import time
	from sample_transforms import *
	
	env_root = '/media/drive3/brdf_project/'
	dataset_root = '/media/drive3/brdf_project/data_generation_pms_semi_cal/'
	
	
	
	
	scene_list = make_scene_list_dicts_all_views(dataset_root,train=False)
	domains_to_load = ['albedo','roughness','normal','depth','mask','env_image']
	size_transform = RandomCropResize((128,128),keys=domains_to_load,list_keys=['images'],scale=(0.25,1))
	scale_transforms = transforms.Compose([NormalizeByMedian(list_keys=['images']),RandomScaleUnique(list_keys=['images'],my_range=(0.2,0.2)),TonemapSRGB(list_keys=['images']),DigitizeRoughness(keys=['roughness']),ErodeMask()])
	
	dataset = MultiIllumDRTorranceDataset(scene_list,domains_to_load=domains_to_load,colocated_as_2=True,transform=transforms.Compose([scale_transforms]),mode='rand2')

	sample = dataset.__getitem__(4)
	
	
	for im_num in range(0,len(sample['images'])):
		plt.figure()
		plt.imshow(sample['images'][im_num])
		print(sample['images'][im_num].dtype)
		#print(im_num,sample['light_dirs'][im_num])
		
	
	

	plt.figure()
	plt.imshow(sample['roughness'].squeeze())
	
	plt.figure()
	plt.imshow((sample['normal']+1)/2)
	plt.figure()
	plt.imshow(sample['mask'].squeeze())
	plt.figure()
	plt.imshow(sample['env_image'].squeeze())
	plt.figure()
	plt.imshow(sample['albedo'].squeeze())
	plt.show()
	
	
	
	
	
	
	
	
	
	
	
	

