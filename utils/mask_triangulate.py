import numpy as np

#mask (m,n)
#faces (k,3), vert_num (m,n), int total_vert
def mask_triangulate(mask):
	vert_num = -np.ones((mask.shape[0]+2,mask.shape[1]+2),dtype=np.int64)
	total_verts = 0
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if mask[i,j]:
				vert_num[i+1,j+1] = total_verts
				total_verts += 1
				
	faces = []
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if vert_num[i+1,j+1] > -1:
			
				if vert_num[i+1,j+2] > -1 and vert_num[i+2,j+1] > -1:
					face = (vert_num[i+1,j+1],vert_num[i+1,j+2],vert_num[i+2,j+1])
					faces.append(face)
			
				if vert_num[i+1,j] > -1 and vert_num[i,j+1] > -1:
					face = (vert_num[i+1,j+1],vert_num[i+1,j],vert_num[i,j+1])
					faces.append(face)
	faces = np.array(faces)
	vert_num = vert_num[1:-1,1:-1]
	return faces, vert_num, total_verts
	
				
