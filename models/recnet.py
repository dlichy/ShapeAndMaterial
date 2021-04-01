import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


def freeze_bn(m):
	if isinstance(m, nn.BatchNorm2d):
		m.eval()

def forgiving_state_restore(net, loaded_dict):
	net_state_dict = net.state_dict()
	new_loaded_dict = {}
	for k in net_state_dict:
		if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
			new_loaded_dict[k] = loaded_dict[k]
			print(k)
		else:
			logging.info('Skipped loading parameter {}'.format(k))
			print('skip: ' + k)
	net_state_dict.update(new_loaded_dict)
	net.load_state_dict(net_state_dict)
	return net

def conv_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		init.xavier_uniform(m.weight, gain=np.sqrt(2))
		#init.normal(m.weight)
		init.constant(m.bias, 0)

	if classname.find('Linear') != -1:
		init.normal(m.weight)
		init.constant(m.bias,1)

	if classname.find('BatchNorm2d') != -1:
		init.normal(m.weight.data, 1.0, 0.2)
		init.constant(m.bias.data, 0.0)

class conv1x1(nn.Module):
	'''(conv => BN => ReLU)'''
	def __init__(self, in_ch, out_ch):
		super(conv1x1, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 1, stride=1,padding=0),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		x = self.conv(x)
		return x

class conv3x3(nn.Module):
	'''(conv => BN => ReLU)'''
	def __init__(self, in_ch, out_ch):
		super(conv3x3, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, stride=2,padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		x = self.conv(x)
		return x

# Define a resnet block
class ResnetBlock(nn.Module):
	def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		super(ResnetBlock, self).__init__()
		self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

	def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		conv_block = []
		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)

		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
					   norm_layer(dim),
					   nn.ReLU(True)]
		if use_dropout:
			conv_block += [nn.Dropout(0.5)]

		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
					   norm_layer(dim)]

		return nn.Sequential(*conv_block)

	def forward(self, x):
		out = x + self.conv_block(x)
		return out
		

	
class RecNetComponent(nn.Module):
	def __init__(self,input_nc, init_feature_nc, output_nc, norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=True):
		super(RecNetComponent,self).__init__()
		
		self.net = nn.Sequential(nn.Conv2d(input_nc, init_feature_nc, kernel_size=7, padding=3,bias=use_bias),
				 norm_layer(init_feature_nc),
				 nn.ReLU(True),
				 ResnetBlock(init_feature_nc,'zero',norm_layer,use_dropout,use_bias),
				 ResnetBlock(init_feature_nc,'zero',norm_layer,use_dropout,use_bias),
				 nn.Conv2d(init_feature_nc, 2*init_feature_nc, kernel_size=3,stride=2, padding=1, bias=use_bias),
				 norm_layer(2*init_feature_nc),
				 nn.ReLU(True),
				 ResnetBlock(2*init_feature_nc,'zero',norm_layer,use_dropout,use_bias),
				 ResnetBlock(2*init_feature_nc,'zero',norm_layer,use_dropout,use_bias),
                                nn.Conv2d(2*init_feature_nc, 4*init_feature_nc, kernel_size=3,stride=2, padding=1,bias=use_bias),
                                norm_layer(4*init_feature_nc),
				 nn.ReLU(True),
				 ResnetBlock(4*init_feature_nc,'zero',norm_layer,use_dropout,use_bias),
				 ResnetBlock(4*init_feature_nc,'zero',norm_layer,use_dropout,use_bias),
				 nn.ConvTranspose2d(4*init_feature_nc,2*init_feature_nc,kernel_size=3, stride=2,padding=1, output_padding=1,bias=use_bias),
			         norm_layer(2*init_feature_nc),
			         nn.ReLU(True),
			         ResnetBlock(2*init_feature_nc,'zero',norm_layer,use_dropout,use_bias),
				 ResnetBlock(2*init_feature_nc,'zero',norm_layer,use_dropout,use_bias),
				 nn.ConvTranspose2d(2*init_feature_nc,init_feature_nc,kernel_size=3, stride=2,padding=1, output_padding=1,bias=use_bias),
			         norm_layer(init_feature_nc),
			         nn.ReLU(True),
			         nn.Conv2d(init_feature_nc, output_nc, kernel_size=7, padding=3)
			         )
			         
	def forward(self, image):
		return self.net(image)


class InitNet(nn.Module):
	def __init__(self,input_nc, init_feature_nc, output_nc, norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=True):
		super(InitNet,self).__init__()
		
		self.net = nn.Sequential(nn.Conv2d(input_nc, init_feature_nc, kernel_size=7, padding=3,bias=use_bias),
				 norm_layer(init_feature_nc),
				 nn.ReLU(True),
				 ResnetBlock(init_feature_nc,'zero',norm_layer,use_dropout,use_bias),
				 ResnetBlock(init_feature_nc,'zero',norm_layer,use_dropout,use_bias),
			         nn.Conv2d(init_feature_nc, output_nc, kernel_size=7, padding=3)
			         )
			         
	def forward(self, image):
		return self.net(image)

def normal_xy_to_normal(normal_xy):
	normal_z = -torch.sqrt( (1-torch.sum(normal_xy**2,dim=1,keepdim=True)).clamp(min=1e-10))
	normal = torch.cat((normal_xy,normal_z),dim=1)
	return normal

class RecNet(nn.Module):
	def __init__(self, input_nc=19, init_feature_nc=64,norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=True):
		super(RecNet, self).__init__()
	
		self.base_net_albedo = InitNet(input_nc,init_feature_nc,3)
		self.base_net_normal = InitNet(input_nc,init_feature_nc,2)
		self.base_net_rough = InitNet(input_nc,init_feature_nc,1)
		
		self.rec_net_albedo = RecNetComponent(input_nc+6,init_feature_nc,3)
		self.rec_net_normal = RecNetComponent(input_nc+6,init_feature_nc,2)
		self.rec_net_rough = RecNetComponent(input_nc+6,init_feature_nc,1)
			         
	#list of images of sizes 32x32,64x64,128x128 ...
	#returns a list with same shape
	def forward(self, images):
		albedo_0 = self.base_net_albedo(images[0])
		normal_0 = self.base_net_normal(images[0])
		rough_0 = self.base_net_rough(images[0])
		
		albedos = [albedo_0]
		normals = [normal_0]
		roughs = [rough_0]

		for i in range(1,len(images)):
			prop = torch.cat((albedos[i-1],normals[i-1],roughs[i-1]),dim=1)
			upsampled_prop = nn.functional.interpolate(prop,scale_factor=2)
			image_and_prop = torch.cat( (images[i],upsampled_prop),dim=1)
			albedo_i = self.rec_net_albedo( image_and_prop )
			normal_i = self.rec_net_normal(image_and_prop)
			rough_i = self.rec_net_rough(image_and_prop)
			
			albedos.append(albedo_i)
			normals.append(normal_i)
			roughs.append(rough_i)
			
		normals = [normal_xy_to_normal(x) for x in normals]
		
		output = {'albedo': albedos, 'normal': normals, 'rough': roughs}
		return output




