from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import glob
import torch
import sys
import scipy.ndimage
import random
import cv2
import warnings
#random.seed(0)



class  RealDataset(Dataset):
	#if i is in to_zero then image i will be zeroed even if it is present in the dataset
	def __init__(self,dataset_root, to_zero, transform=None):
		scene_paths = glob.glob(os.path.join(dataset_root,'*'))
		scene_list = []
		
		for scene_path in scene_paths:
			scene = {}
			scene['name'] = os.path.basename(scene_path)
			
			mask_path = glob.glob(os.path.join(scene_path,'mask*'))
			if len(mask_path) > 1:
				raise Exception('multiple masks found in scene {}'.format(scene_path))
			if len(mask_path) == 0:
				warnings.warn('warning: no mask found in scene {} using all ones as mask'.format(scene_path))
				scene['mask_path'] = None
			else:
				scene['mask_path'] = mask_path[0]
			
			some_image_found = False
			image_paths = []
			for i in range(0,6):
				im_path = glob.glob(os.path.join(scene_path,'{}*'.format(i)))
				if len(im_path) > 1:
					raise Exception('multiple images named {} in folder {}'.format(i,scene_path))
				if len(im_path) == 1:
					image_paths.append(im_path[0])
					some_image_found = True
				elif len(im_path) == 0: 
					if i == 2:
						warnings.warn('warning no image 2 (colocated light), bad performance expected')
					image_paths.append(None)
			scene['image_paths'] = image_paths
			if some_image_found:
				scene_list.append(scene)
			else:
				warnings.warn('warning no images found in {}. Skipping'.format(scene_path))	
		
		self.to_zero = to_zero
		self.scene_list = scene_list
		self.transform = transform
    
	def __len__(self):
		return len(self.scene_list)

	
	def __getitem__(self, idx):
		scene = self.scene_list[idx]
		sample = {}
		sample['name'] = scene['name']
		
		images = []
		
		for im_path in scene['image_paths']:
			if im_path is not None:
				image = np.array(Image.open(im_path)).astype(np.float32) / 255
				image_size = image.shape
				images.append(image)
			else:
				images.append(None)
	
		#replace missing images with zeros
		images = [np.zeros(image_size,dtype=np.float32) if im is None else im for im in images]	
		#zero images
		for i in self.to_zero:
				images[i] = 0*images[i]
				
				
				
		if scene['mask_path'] is not None:
			mask = np.array(Image.open(scene['mask_path'])).astype(np.float32) / 255
			if mask.ndim > 2:
				mask = mask[:,:,0:1]
			else:
				mask = np.expand_dims(mask,2)
		else:
			mask = np.ones((*image_size[0:2],1),dtype=np.float32)
		

		
		
		sample['images'] = images
		sample['mask'] = mask
		if self.transform:
			self.transform(sample)
		
		return sample
		
