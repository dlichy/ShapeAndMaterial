import torch

def save_checkpoint(save_path, net=None, optimizer=None, epoch=0):
	checkpoint = {}
	checkpoint['epoch'] = epoch
	if net is not None:
		checkpoint['state'] = net.module.state_dict()
	if optimizer is not None:
		checkpoint['optimizer'] = optimizer.state_dict()
	torch.save(checkpoint, save_path)
	
def load_checkpoint(load_path, net=None, optimizer=None, device=None):
	checkpoint = torch.load(load_path, map_location = torch.device(device))
	if net is not None:
		net.load_state_dict(checkpoint['state'])
	if optimizer is not None:
		if 'optimizer' in checkpoint.keys():
			optimizer.load_state_dict(checkpoint['optimizer'])
	return checkpoint['epoch']
