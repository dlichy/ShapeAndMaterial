import numpy as np

def linear_to_srgb(image,clip=True):
	image = image.clip(min=0) # for some reason images can have very small negative values
	out = np.where(image < 0.0031308, 12.92*image, 1.055*np.power(image,1/2.2) - 0.055)
	
	if clip:
		out=out.clip(0,1.0)
	return out
	
def srgb_to_linear(image):
	return np.where(image < 0.04045, image/12.92, np.power((image+0.055)/1.055, 2.4))	
