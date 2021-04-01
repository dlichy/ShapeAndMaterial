import numpy as np

#vert_props is list of length len(prop_names) all numpy arrays of length N. faces is Mx3

def write_ply(save_name,vert_props,prop_names=['x','y','z','nx','ny','nz'],prop_types=['float32' for _ in range(0,6)],faces=None):
	num_verts = len(vert_props[0])
	if faces is not None:
		num_faces = faces.shape[0]
	else:
		num_faces = 0
		
	with open(save_name,'w') as f:
		#write header
		f.write("ply \n format ascii 1.0 \n")
		f.write("element vertex {} \n".format(num_verts))

		for dtype, name in zip(prop_types, prop_names):
			f.write("property {} {} \n".format(dtype,name))
		
		if num_faces > 0:
			f.write("element face {} \n".format(num_faces))
			f.write("property list uchar int vertex_indices \n")
		f.write("end_header \n")
		
		#write vert properties
		for i in range(0,vert_props[0].shape[0]):
			for j in range(0,len(prop_names)):
				f.write("{} ".format(vert_props[j][i]))
			f.write("\n")
				
		#write faces
		for i in range(0,num_faces):
			f.write("{} {} {} {} \n".format(3,faces[i,0], faces[i,1], faces[i,2]))

			
	
if __name__ == '__main__':
	pc = np.random.randn(5,3)
	*vert_props, = pc.T
	prop_types = ['float32' for _ in range(0,3)]
	print(vert_props)
	print(prop_types)
	#write_ply('test.ply',vert_props,prop_names=['x','y','z'],prop_types=prop_types)

	#23
	#01
	x = np.array([0,1,0,1])
	y = np.array([0,0,1,1])
	z = np.array([0,0,0,0])
	
	faces=np.array([[0,1,2],[1,3,2]])
	nx = np.array([0,0,0,0])
	ny = np.array([0,0,0,0])
	nz = np.array([1,1,1,1])
	
	write_ply('test_square.ply',[x,y,z,nx,ny,nz],faces=faces)
