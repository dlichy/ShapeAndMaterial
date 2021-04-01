import torch
import torch.nn.functional as F

def make_scales(image,normalized=False,mode='bilinear',min_size=32):
	if normalized:
		image = F.normalize(image,dim=1,p=2)
		
	images = [image]
	while image.size(2) > min_size:
		image = F.interpolate(image,scale_factor=0.5,mode=mode,align_corners=False)
		if normalized:
			image = F.normalize(image,dim=1,p=2)
		images.append(image)
	images.reverse()
	return images
