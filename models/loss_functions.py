import torch
import numpy as np


#bxcxmxn, bx1xmxn
#returns b
def masked_mean(x,mask):
        mask = mask > 0.5
        num = torch.sum( (x*mask).flatten(1),dim=1)
        denom = torch.sum(mask.flatten(1),dim=1)
        return num/denom.clamp(min=1.0)


def angular_err(a,b,mask):
	angular_error = torch.acos(torch.sum(a*b,dim=1,keepdim=True).clamp(-1,1))
	angular_error_deg = (180.0 / np.pi)*angular_error
	mae_per_item = masked_mean(angular_error_deg,mask)
	mae = torch.mean(mae_per_item)
	return mae
	
def L1_loss(a,b,mask):
	loss_per_item = masked_mean(torch.abs(a-b),mask)
	loss = torch.mean(loss_per_item)
	return loss
	
def angular_err_percent(a,b,mask,thresh_angles):
	mask = mask > 0.5
	angular_error = torch.acos(torch.sum(a*b,dim=1,keepdim=True).clamp(-1,1))
	angular_error_deg = (180.0 / np.pi)*angular_error
	angular_error_deg[~mask] = 10000.0
	angular_error_deg = angular_error_deg.flatten(1)
	
	total_pix = torch.sum(mask.flatten(1),dim=1)

	percentages = []
	for ang in thresh_angles:
		pix_less_than = torch.sum(angular_error_deg<ang,dim=1).float()
		percentages.append(pix_less_than/total_pix)
	return percentages	


