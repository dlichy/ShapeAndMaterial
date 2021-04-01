import torch
from utils.make_scales import make_scales
from models.loss_functions import angular_err

def prepare_data(sample,device):
	images = sample['images']
	mask = sample['mask']
	images.append(mask)
	images = torch.cat(images,dim=1).to(device)
	images_scale = make_scales(images)
	
	target={}
	target['mask'] = make_scales(mask.to(device))
	
	keys = sample.keys()
	if 'albedo' in keys:
		albedo = sample['albedo'].to(device)
		target['albedo'] = make_scales(albedo)
	if 'normal' in keys:
		normal = sample['normal'].to(device)
		target['normal'] = make_scales(normal)
	if 'rough' in keys:
		rough = sample['rough'].to(device)
		target['rough'] = make_scales(rough)
	
	return images_scale, target


def run_epoch(net, train, dataloader, device, criterion=None, optimizer=None, scalar_logger=None, image_logger=None):
	
	#if mode == train ensure has criterion and optimizer
	if train:
		net.train()
	else:
		net.eval()

	for batch_num, sample in enumerate(dataloader):
		
		images, target = prepare_data(sample,device)

		if train:
			optimizer.zero_grad()
		
		output = net(images)

		if criterion is not None:
			loss,losses_dict = criterion(output,target)
		
		if train:
			loss.backward()
			optimizer.step()
			
		if 'normal' in target.keys():
			m_ang_err = angular_err(output['normal'][-1],target['normal'][-1],target['mask'][-1])
			losses_dict['angular_error'] = m_ang_err.item()
		
		if scalar_logger is not None:
			scalar_logger(losses_dict,sample['name'][0])
			
		if image_logger is not None:
			log_images = {}
			for k,v in output.items():
				log_images['output_' + k] = v
			for k,v in target.items():
				log_images['target_' + k] = v 
			log_images['images'] = images
			image_logger(log_images,sample['name'][0])
			
