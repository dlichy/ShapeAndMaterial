from utils.hb_algorithm import *
from utils.mask_triangulate import *
from utils.write_ply import *
from PIL import Image
import torch
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('input_paths', help='csv of paths to input format of each lines is normal_path, mask_path, albedo_path, rough_path, output_path')
parser.add_argument('--mask_erosion', type=int, default=6, help='how much to erodes mask')
parser.add_argument('--optimizer', choices=['LBFGS','Adam'], default='LBFGS', help='optimizer for normal integration')
parser.add_argument('--gpu', default=False,  action='store_true', help='enable to run on gpu')
parser.add_argument('--iters', type=int, default=150, help='iterations for normal integration')
opt = parser.parse_args()

#channel_names is only used if reading exr images
def load_image(im_path, channel_names=['R','G','B']):
	if not os.path.exists(im_path):
		return None
	if os.path.splitext(im_path)[1]=='.exr':
		from utils.exr_read_write_funs import read_exr
		image = read_exr(im_path, channel_names=channel_names)
	else:
		image = np.array(Image.open(im_path))/255
		image = image.astype(np.float32)
	return image


if opt.gpu:
	device = 'cuda'
else:
	device = 'cpu'
	
erosion_size = (opt.mask_erosion,opt.mask_erosion)

flip = torch.tensor([-1,1,-1.0]).view(1,3,1,1)

#parse and verify inputs
all_paths = []
with open(opt.input_paths,'r') as f:
	for line in f.readlines():
		line = line.strip()
		if line == '':
			continue
		paths = line.split(',')
		varified_paths = []
		if len(paths) != 5:
			raise Exception('all lines in input csv must have exactly 4 commas')
		#normal and mask are required
		for p in paths[:2]:
			p = p.strip()
			if not os.path.exists(p):
				raise Exception('file not found: {}'.format(p))
			else:
				varified_paths.append(p)
		#albedo and roughness are optional
		for p in paths[2:4]:
			p = p.strip()
			if os.path.exists(p):
				varified_paths.append(p)
			else: 
				varified_paths.append(None)
		#if output path provided use it, else save in location of normals
		p = paths[4].strip()
		if os.path.exists(p):
			print('p ', p)
			varified_paths.append(p)
		else:
			output_path = os.path.join( os.path.split(paths[0].strip())[0], 'mesh.ply')
			varified_paths.append(output_path)
			
		all_paths.append(varified_paths)



for normal_path, mask_path, albedo_path, rough_path, output_path in all_paths:
	normal_np = load_image(normal_path)
	mask_np = load_image(mask_path,channel_names=['Y'])
	albedo_np = load_image(albedo_path)
	rough_np = load_image(rough_path,channel_names=['Y'])
	
	
	normal_np = 2*normal_np - 1
	mask_np = mask_np > 0.5

	mask_np = ndimage.binary_erosion(mask_np.squeeze(), structure=np.ones(erosion_size))

	normal = torch.nn.functional.normalize(torch.from_numpy(normal_np).permute([2,0,1]).unsqueeze(0),dim=1).float()*flip
	mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)


	normal = normal.to(device)
	mask = mask.to(device)
	
	recon_depth, recon_normal, pc = integrate(normal,mask,optimizer=opt.optimizer,iters=opt.iters)
	
	pc_np = pc.view(3,-1).t().cpu().numpy()
	
	faces, vert_idx, _ = mask_triangulate(mask_np)
	verts = pc_np[mask_np.reshape(-1),:]
	normal = normal.cpu().squeeze(0).view(3,-1).t().numpy()[mask_np.reshape(-1),:]
	
	
	props = [verts[:,0],verts[:,1],verts[:,2], normal[:,0], normal[:,1], normal[:,2]]
	prop_names = ['x','y','z','nx','ny','nz']
	prop_types = ['float32' for _ in range(0,6)]
	
	#has albedo
	if albedo_path is not None:
		albedo = albedo_np.reshape(-1,3)[mask_np.reshape(-1),:]
		props += [albedo[:,0],albedo[:,1],albedo[:,2]]
		prop_names += ['albedo_r','albedo_g','albedo_b']
		prop_types +=  ['float32' for _ in range(0,3)]
	
	#has rough
	if rough_path is not None:
		rough = rough_np.reshape(-1)[mask_np.reshape(-1)]
		props += [rough]
		prop_names += ['rough_r']
		prop_types +=  ['float32']
	
	write_ply(output_path,props,prop_names=prop_names,prop_types=prop_types,faces=faces)



