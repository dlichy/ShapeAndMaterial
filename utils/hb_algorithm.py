import numpy as np
import torch
import torch.optim as optim
from PIL import Image
import scipy.ndimage as ndimage
	
#bxcxmxn
def fd(image,axis):
	c = image.size(1)
	if axis == 'u':
		filt = torch.tensor([[0.0,0,0],[0,-1,1],[0,0,0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1)
	elif axis=='v':
		filt = torch.tensor([[0.0,0,0],[0,-1,0],[0,1,0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1)
	return  torch.nn.functional.conv2d(image,filt,padding=1,groups=c)	
	
def avg(image,axis):
	c = image.size(1)
	if axis == 'u':
		filt = 0.5*torch.tensor([[0.0,0,0],[0,1,1],[0,0,0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1)
	elif axis=='v':
		filt = 0.5*torch.tensor([[0.0,0,0],[0,1,0],[0,1,0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1)
	return  torch.nn.functional.conv2d(image,filt,padding=1,groups=c)	
	
#bx1xmxn
def make_omegas(omega):
	omega1 = omega & torch.roll(omega,-1,dims=3)
	omega2 = omega & torch.roll(omega,-1,dims=2)
	return omega1, omega2
	
	
	
def hb_algorithm(p,q,omega,delta,init_z=None,iters=1000):
	omega1,omega2 = make_omegas(omega)
	
	p_u = avg(p,'u')
	q_v = avg(q,'v')
	
	if not init_z == None:
		z = init_z.clone() 
	else:
		z = torch.zeros_like(p)
	z.requires_grad = True
	
	optimizer = optim.Adam([z],lr=0.0005)
	
	for i in range(0,iters):
		optimizer.zero_grad()
		dz_du = fd(z,'u')/delta
		dz_dv = fd(z,'v')/delta
		
		temp1 = dz_du-p_u
		loss1 = torch.sum(temp1[omega1]**2)
		
		temp2 = dz_dv-q_v
		loss2 = torch.sum(temp2[omega2]**2)
		
		loss = loss1+loss2
		print('iter: ', i, ' loss: ', loss.item())
		loss.backward()
		optimizer.step()
	return z
	

def hb_algorithm_LBFGS(p,q,omega,delta,init_z=None,iters=1000):
	omega1,omega2 = make_omegas(omega)
	
	p_u = avg(p,'u')
	q_v = avg(q,'v')
	
	if not init_z == None:
		z = init_z.clone() 
	else:
		z = torch.zeros_like(p)
	z.requires_grad = True
	
	optimizer = optim.LBFGS([z], line_search_fn="strong_wolfe")
	
	for i in range(0,iters):
		def get_loss():
			optimizer.zero_grad()
			dz_du = fd(z,'u')/delta
			dz_dv = fd(z,'v')/delta
		
		
			temp1 = dz_du-p_u
			loss1 = torch.sum(temp1[omega1]**2)
		
			temp2 = dz_dv-q_v
			loss2 = torch.sum(temp2[omega2]**2)
		
			loss = loss1+loss2
			
			loss.backward()
			return loss
		loss = optimizer.step(get_loss)
		print('iter: ', i, ' loss: ', loss.item())
	return z
	
	
	
def integrate(normal,mask,optimizer='LBFGS',iters=100):
	#make left hand side
	denom = normal[:,2,:,:].clamp(max=-0.01)
	p = (normal[:,0,:,:]/denom)
	q = (normal[:,1,:,:]/denom)

	delta = 2/normal.size(2)

	p = p.unsqueeze(1).to(normal.device)
	q = q.unsqueeze(1).to(normal.device)
	
	if optimizer == 'LBFGS':
		recon_depth = hb_algorithm_LBFGS(p,q,mask,delta,None,iters=iters).cpu().detach()
	elif optimizer == 'Adam':
		recon_depth = hb_algorithm(p,q,mask,delta,None,iters=iters).cpu().detach()
	else:
		raise Exception('optimizer_not_recognized')
	
	nx = -fd(recon_depth,'u')/delta
	ny = -fd(recon_depth,'v')/delta
	nz = -torch.ones_like(recon_depth)
	
	recon_normal = torch.nn.functional.normalize(torch.cat((nx,ny,nz),dim=1),dim=1)
	
	x_pix = torch.linspace(1,-1,normal.size(2))
	y_pix = torch.linspace(1,-1,normal.size(2))
	y,x = torch.meshgrid([y_pix,x_pix])
	
	pc = torch.stack((x,y,recon_depth.squeeze()),dim=0)

	return recon_depth, recon_normal, pc


	

	

