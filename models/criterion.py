import torch
import torch.nn as nn
from . import loss_functions

class Criterion(nn.Module):
	def __init__(self, normal_weight=1, albedo_weight=1, rough_weight=1, record_scale_loss=True, normal_loss_xy_only=True):
    		super(Criterion,self).__init__()
    		
    		self.normal_weight = normal_weight
    		self.albedo_weight = albedo_weight
    		self.rough_weight = rough_weight
    		self.record_scale_loss = record_scale_loss
    		self.normal_loss_xy_only = normal_loss_xy_only
    
	def list_L1_loss(self,a_s,b_s,masks):
		losses = []
		loss = 0
		for a,b,mask in zip(a_s,b_s,masks):
			loss_i = loss_functions.L1_loss(a,b,mask)
			loss = loss + loss_i
			losses.append(loss_i)
	
		return loss, losses

	#for training output and target should have a list of normals, albedos, and roughnesses
	def forward(self, output, target):
		loss = 0
		loss_dict = {}
        	
		target_keys = target.keys()
        	
		if 'normal' in target_keys:
			if self.normal_loss_xy_only:
        			normal_out = [x[:,0:2,:,:] for x in output['normal']]
        			normal_target = [x[:,0:2,:,:] for x in target['normal']]
			else:
				normal_out = output['normal']
				normal_target = target['normal']
        			
			n_loss, n_loss_scales = self.list_L1_loss(normal_out,normal_target,target['mask'])		
			loss += self.normal_weight*n_loss
			loss_dict['normal_loss'] = n_loss.item()
			if self.record_scale_loss:
				loss_dict['normal_loss_scale'] = [x.item() for x in n_loss_scales]
         	
		if 'albedo' in target_keys:
			a_loss, a_loss_scales = self.list_L1_loss(output['albedo'],target['albedo'],target['mask'])		
			loss += self.albedo_weight*a_loss
			loss_dict['albedo_loss'] = a_loss.item()
			if self.record_scale_loss:
				loss_dict['albedo_loss_scale'] = [x.item() for x in a_loss_scales]
         	
		if 'rough' in target_keys:
			r_loss, r_loss_scales = self.list_L1_loss(output['rough'],target['rough'],target['mask'])		
			loss += self.rough_weight*r_loss
			loss_dict['rough_loss'] = r_loss.item()
			if self.record_scale_loss:
				loss_dict['rough_loss_scale'] = [x.item() for x in r_loss_scales]
		
		if isinstance(loss, torch.Tensor):
			loss_dict['loss'] = loss.item()
		else: 
			loss_dict['loss'] = loss
			       
		return loss, loss_dict
     
