import numpy as np
import torch
import torchvision
from torchvision import transforms
import scipy.ndimage
import random
import cv2
from . import sRGB_funs

random.seed(0)


class ErodeMask:
	def __init__(self,mask_erosion_size=(6,6)):
		self.mask_erosion_size = mask_erosion_size
	
	def __call__(self,sample):
		mask = sample['mask']
		mask = mask.squeeze() > 0
		mask_eroded = scipy.ndimage.binary_erosion(mask,structure=np.ones(self.mask_erosion_size))
		sample['mask'] = np.expand_dims(mask_eroded.astype('single'),axis=2)
		return sample
		

class MyToTensor:
	def __init__(self,keys=[],list_keys=[]):
		self.toTensor = transforms.ToTensor()
		self.keys = keys
		self.list_keys = list_keys
		
	def __call__(self,sample):
		for key in self.keys:
			sample[key] = self.toTensor(sample[key])
		
		for list_key in self.list_keys:
			for i,image in enumerate(sample[list_key]):
				sample[list_key][i] = self.toTensor(image)
			
		return sample

#randomly crop a region of size X*image_size where X is a random number between scale[0] and scale[1]
class RandomCropResize:
	def __init__(self, output_size, keys=[], list_keys=[], scale=(0.5,1), ratio=(1,1)):
		self.scale = scale
		self.ratio = ratio
		self.output_size = output_size
		self.keys = keys
		self.list_keys = list_keys
		
	def __call__(self,sample):
		if len(self.keys) > 0:
			s = sample[self.keys[0]].shape
		else: 
			s = sample[self.list_keys[0]][0].shape
		
		input_width = s[1]
		input_height = s[0]
		
		aspect_ratio = random.uniform(*self.ratio)
		width = int(input_width*random.uniform(*self.scale))
		height = int(aspect_ratio*width)
		
		top = random.randrange(0,input_height-height+1)
		left = random.randrange(0,input_width-width+1)
		
		
		for key in self.keys:
			cropped = sample[key][top:(top+height),left:(left+width),:]
			sample[key] = cv2.resize(cropped,dsize=self.output_size,interpolation = cv2.INTER_LINEAR)
		
		for list_key in self.list_keys:
			for i,image in enumerate(sample[list_key]):
				cropped = image[top:(top+height),left:(left+width),:]
				sample[list_key][i] = cv2.resize(cropped,dsize=self.output_size,interpolation = cv2.INTER_LINEAR)
		
		return sample
		
#effectively make features smaller
class ShrinkAndPad:
	def __init__(self,keys=[], list_keys=[], my_range=(0.4,1)):
		self.range = my_range
		self.keys = keys
		self.list_keys = list_keys
		
	def __call__(self,sample):
		if len(self.keys) > 0:
			s = sample[self.keys[0]].shape
		else: 
			s = sample[self.list_keys[0]][0].shape
	
		scale = random.uniform(*self.range)
		original_size = np.array(s[0:2])
		new_size = (scale*original_size).astype('int')
		pad_size = original_size-new_size
		
		tl_pad = (np.random.rand((2))*(pad_size+1)).astype('int')
		br_pad = pad_size - tl_pad
		
		pad_shape = ((tl_pad[0],br_pad[0]),(tl_pad[1],br_pad[1]),(0,0))
		
		for key in self.keys:
			resized = cv2.resize(sample[key].squeeze(),(new_size[0],new_size[1]),interpolation = cv2.INTER_LINEAR)
			if resized.ndim == 2:
				resized = np.expand_dims(resized,2)
			sample[key] = np.pad(resized,pad_shape)
			
		for list_key in self.list_keys:
			for i,image in enumerate(sample[list_key]):
				resized=cv2.resize(image.squeeze(),(new_size[0],new_size[1]),interpolation = cv2.INTER_LINEAR)
				if resized.ndim == 2:
					resized = np.expand_dims(resized,2)
				sample[list_key][i] = np.pad(resized,pad_shape)
				
		return sample
		
class RandomChooseTransform:
	def __init__(self,transforms,probs):
		self.probs = probs
		self.transforms = transforms
		
	def __call__(self,sample):
		transform = random.choices(self.transforms,weights=self.probs)[0]
		return transform(sample)


class RemoveNans:
	def __init__(self,keys=[],list_keys=[]):
		self.keys = keys
		self.list_keys = list_keys
		
	def __call__(self,sample):
		for key in self.keys:
			image =  sample[key]
			image[~np.isfinite(image)] = 0
			sample[key] = image
		
		for list_key in self.list_keys:
			for i,image in enumerate(sample[list_key]):
				image[~np.isfinite(image)] = 0
				sample[list_key][i] = image
		
		return sample


#randomly scale all images by a single random number sampled uniformly in my_range
class RandomScale:
	def __init__(self,keys=[], list_keys=[],my_range=(0.6,1.4)):
		self.range = my_range
		self.keys = keys
		self.list_keys
		
	def __call__(self,sample):
		scale = random.uniform(*self.range)
		for key in self.keys:
			sample[key] = scale*sample[key]
		
		for list_key in self.list_keys:
			for i,image in enumerate(sample[list_key]):
				sample[list_key][i] = scale*image
				 
		return sample
	
#randomly scale each image by a unique random number random number sampled uniformly in my_range	
class RandomScaleUnique:
	def __init__(self,keys=[], list_keys=[],my_range=(0.3,1.8)):
		self.range = my_range
		self.keys = keys
		self.list_keys = list_keys
		
	def __call__(self,sample):
		for key in self.keys:
			scale = random.uniform(*self.range)
			sample[key] = scale*sample[key]
		
		for list_key in self.list_keys:
			for i,image in enumerate(sample[list_key]):
				scale = random.uniform(*self.range)
				sample[list_key][i] = scale*image
				 
		return sample
		
	
#scale images so they have median 1
class NormalizeByMedian:
	def __init__(self,keys=[],list_keys=[]):
		self.keys = keys
		self.list_keys = list_keys
	
	def __call__(self,sample):
		for key in self.keys:
			image = sample[key]
			scale = 1/(np.median(image).clip(min=0.001))
			sample[key] = scale*image
		
		for list_key in self.list_keys:
			for i,image in enumerate(sample[list_key]):
				scale = 1/(np.median(image).clip(min=0.001))
				sample[list_key][i] = scale*image

		return sample



class RandomNoise:
	def __init__(self,keys=[],list_keys=[],my_range=(-0.025,0.025)):
		self.keys = keys
		self.list_keys = list_keys
		self.range = my_range
	
	def __call__(self,sample):
		for key in self.keys:
			image = sample[key]
			noise = np.random.uniform(*self.range,image.shape).astype(np.float32)
			sample[key] = image + noise
		
		for list_key in self.list_keys:
			for i,image in enumerate(sample[list_key]):
				noise = np.random.uniform(*self.range,image.shape).astype(np.float32)
				sample[list_key][i] = image + noise
		return sample
	

#SRGB tonemap sample
class TonemapSRGB:
	def __init__(self,keys=[],list_keys=[],clip=True):
		self.keys = keys
		self.list_keys = list_keys
		self.clip = clip
		
	def __call__(self,sample):
		for key in self.keys:
			sample[key] = sRGB_funs.linear_to_srgb(sample[key],self.clip)
		
		for list_key in self.list_keys:
			for i,image in enumerate(sample[list_key]):
				sample[list_key][i] = sRGB_funs.linear_to_srgb(image,self.clip)

		return sample
		
class PadToSquare:
	def __init__(self,keys=[],list_keys=[],record_pad_shape=False):
		self.record_pad_shape = record_pad_shape
		self.keys = keys
		self.list_keys = list_keys
	
	def __call__(self,sample):
		if len(self.keys) > 0:
			s = sample[self.keys[0]].shape
		else: 
			s = sample[self.list_keys[0]][0].shape
	
		if s[0] < s[1]:
			top_pad = int((s[1]-s[0])/2)
			bottom_pad = s[1]-s[0] - top_pad
			pad_shape = ((top_pad,bottom_pad),(0,0),(0,0))
		else:
			left_pad = int((s[0]-s[1])/2)
			right_pad = s[0]-s[1] - left_pad
			pad_shape = ((0,0),(left_pad,right_pad),(0,0))
        	
		
		for key in self.keys:
			sample[key] = np.pad(sample[key],pad_shape)
			
		for list_key in self.list_keys:
			for i,image in enumerate(sample[list_key]):
				sample[list_key][i] = np.pad(image,pad_shape)
		
		if self.record_pad_shape:
			sample['pad_shape'] = np.array(pad_shape)
		
		return sample
		
		
class PadSquareToPower2:
	def __init__(self,keys=[],list_keys=[],record_pad_shape=False):
		self.record_pad_shape = record_pad_shape
		self.keys = keys
		self.list_keys = list_keys
	
	def __call__(self,sample):
		if len(self.keys) > 0:
			s = sample[self.keys[0]].shape[0]
		else: 
			s = sample[self.list_keys[0]][0].shape
	
		new_size = 2**np.ceil(np.log2(s))
			

		top_pad = int((new_size-s)/2)
		bottom_pad = int(new_size-s - top_pad)
		pad_shape = ((top_pad,bottom_pad),(top_pad,bottom_pad),(0,0))

		for key in self.keys:
			sample[key] = np.pad(sample[key],pad_shape)
			
		for list_key in self.list_keys:
			for i,image in enumerate(sample[list_key]):
				sample[list_key][i] = np.pad(image,pad_shape)
		
		if self.record_pad_shape:
			sample['pad_shape'] += np.array(pad_shape)
		
		return sample
        

class Resize:
	def __init__(self,size,keys=[],list_keys=[]):
		self.keys = keys
		self.list_keys = list_keys
		self.size = size

	def __call__(self,sample):
		
		for key in self.keys:
			image = cv2.resize(sample[key],self.size, interpolation=cv2.INTER_LINEAR)
			if image.ndim == 2:
				image = np.expand_dims(image,2)
			sample[key] = image
					
		for list_key in self.list_keys:
			for i,image in enumerate(sample[list_key]):
				image = cv2.resize(image,self.size, interpolation=cv2.INTER_LINEAR)
				if image.ndim == 2:
					image = np.expand_dims(image,2)
				sample[list_key][i] = image
				
		return sample
		


