import OpenEXR
import Imath
import numpy as np

def read_exr(file_path,channel_names=['R','G','B'], dtype=np.float16):
	image = OpenEXR.InputFile(file_path)
	dw = image.header()['dataWindow']
	size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
	string_channels = image.channels(channel_names)
	float_channels = []
	for s in string_channels:
		c = np.fromstring(s, dtype = dtype)
		c.shape = (size[1],size[0],1)
		float_channels.append(c)
	out_image = np.concatenate(float_channels,axis=2)	
	return out_image.astype(np.float32)

def write_exr(image, save_path, channel_names=['R','G','B']):
	size = image.shape
	HEADER = OpenEXR.Header(size[1],size[0])
	half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
	HEADER['channels'] = dict([(c, half_chan) for c in channel_names])
	
	pixels_to_write = {}
	for i,c in enumerate(channel_names):
		pixels_to_write[c] = image[:,:,i].astype(np.float16).tostring()

	exr = OpenEXR.OutputFile(save_path, HEADER)
	exr.writePixels(pixels_to_write)
	exr.close()
