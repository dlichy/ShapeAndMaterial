from collections import defaultdict
import os
from PIL import Image
import torch
import numpy as np

class ScalarLogger:
	
	def __init__(self, save_file, log_freq=50, keys=[]):
		
		self.counter = 0
		self.save_file = save_file
		self.log_freq = log_freq
		self.keys = keys
		
		#dict with keys=scalar_keys and values = sum of losses until that point
		self.log_since_last = defaultdict(lambda: 0)
		#dict with keys=scalar_keys and values = sum of losses until that point
		self.log_all = defaultdict(lambda: 0)
		
		self.log_string = ''
			
	#data is a dict where values are floats or list of floats
	def __call__(self, data, other_text=''):
		for key in self.keys:
			if not isinstance(data[key], list):
				self.log_all[key] += data[key]
				self.log_since_last[key] += data[key]
			else:
				for i,v in enumerate(data[key]):
					self.log_all[key + ('_%d'%i)] += v
					self.log_since_last[key + ('_%d'%i)] += v
		self.counter +=1
		if self.counter % self.log_freq == 0:
			string = 'iter {} - {} \n'.format(self.counter-self.log_freq,self.counter-1)
			for k, v in sorted(self.log_since_last.items()):
				string += '{}: {} \n'.format(str(k), str(v/self.log_freq))
			string += 'other text: {} \n'.format(other_text)
			print(string)
			self.log_string += string
			self.log_since_last = defaultdict(lambda: 0)
    		    	
	def summarize(self):
		string = '----------  Epoch Summary -----------\n'
		for k, v in sorted(self.log_all.items()):
			string += '{}: {} \n'.format(str(k), str(v/self.counter))
		print(string)
		self.log_string += string
		with open(self.save_file,'w') as f:
			f.write(self.log_string)
			

class NormalScale:
	def __init__(self):
		self.flip = torch.tensor([-1,1,-1.0]).view(1,3,1,1)
		
	def __call__(self,sample):
		for key in sample.keys():
			if 'normal' in key:
				if not isinstance(sample[key], list):
					im = sample[key]
					sample[key] = (im*self.flip.to(im.device)+1)/2
				else: 
					for i,im in enumerate(sample[key]):
						sample[key][i] = (im*self.flip.to(im.device)+1)/2
		return sample
		
class ImageLogger:
	# keys = None saves all images in the dict
	# indices_to_save = a list of indices in the batch to save. None saves all images in the batch
	def __init__(self,save_root, log_freq=50, keys=None, save_ext='.png', transform=NormalScale(), indices_to_save=None, save_csv_for_integration=False,save_last_only=False):
		if save_ext == '.exr':
			from .exr_read_write_funs import write_exr
			self.my_write_exr = write_exr

		if not os.path.exists(save_root):
			os.mkdir(save_root)
			
		self.counter = 0
		self.log_freq = log_freq
		self.save_root = save_root
		self.save_ext = save_ext
		self.transform = transform
		self.keys = keys
		self.indices_to_save = indices_to_save
		self.save_last_only = save_last_only
		self.save_csv_for_integration=save_csv_for_integration
		
		if self.save_csv_for_integration:
			self.csv_text = ''
			
	def write_csv(self):
		if not self.save_csv_for_integration:
			Exception('trying to write csv but save_csv_for_integration=False')
		with open(os.path.join(self.save_root, 'integration_data.csv'),'w') as f:
			f.write(self.csv_text)
			
	
	#image is tensor of size (1|3,m,n)
	def save_image(self, save_name, image):
		save_name = save_name + self.save_ext
		image = image.permute([1,2,0]).detach().cpu().numpy()
		
		if self.save_ext == '.exr':
			if image.shape[2] == 1:
				channel_names = ['Y']
			elif image.shape[2] == 3:
				channel_names = ['R','G','B']
			self.my_write_exr(image, save_name, channel_names)
		else:
			pil_image = Image.fromarray(np.uint8(255*image.squeeze().clip(0,1)))
			pil_image.save(save_name)
		
	def save_image_batch(self, save_name, image):
		#determine which images of the batch to save
		if self.indices_to_save == None:
			indices = [x for x in range(image.size(0))]
		else:
			indices = [x for x in self.indices_to_save if x < image.size(0)]
			
		ch = image.size(1)
		for b in indices:
			k = 0		
			while k+3 <= ch:
				self.save_image(save_name + 'b{}_ch{}'.format(b,k), image[b,k:k+3,:,:])
				k = k+3
			while k < ch:
				self.save_image(save_name + 'b{}_ch{}'.format(b,k), image[b,k:k+1,:,:])
				k = k+1
		
		
	
	#values for image_dict are either tensors of shape (b,k,m,n) or lists of (b,k,m,n)
	#list of folder names to save images under
	def __call__(self, image_dict, folder_name):
		self.counter +=1
		if self.counter % self.log_freq == 0:
			folder_name = os.path.join(self.save_root, folder_name)
			if not os.path.exists(folder_name):
				os.mkdir(folder_name)
		
			if self.transform is not None:
				image_dict = self.transform(image_dict)
			
			if self.keys is not None:
				curr_keys = self.keys
			else:
				curr_keys = image_dict.keys()

			for key in curr_keys:
				if not isinstance(image_dict[key], list):
					self.save_image_batch(os.path.join(folder_name,key),image_dict[key])
				else:
					if self.save_last_only:
						v = image_dict[key][-1]
						i = len(image_dict[key])-1
						self.save_image_batch(os.path.join(folder_name,key + ('_s%d_'%i)),v)
					else:
						for i,v in enumerate(image_dict[key]):
							self.save_image_batch(os.path.join(folder_name,key + ('_s%d_'%i)),v)
		
			
			if self.save_csv_for_integration:
				for k in ['output_normal','target_mask','output_albedo','output_rough']:
					self.csv_text += os.path.join(folder_name,'%s_s%d_b0_ch0%s'%(k,len(image_dict[k])-1,self.save_ext))
					self.csv_text+= ','
				self.csv_text += '\n'
					
		
			
	
	
