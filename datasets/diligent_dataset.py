import os
import numpy as np
import scipy.io as sio
from PIL import Image
import cv2
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
np.random.seed(0)



class DiligentDataset(data.Dataset):
    def __init__(self, dataset_root, select_idx=range(0,96), transform=None):
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')
        self.root  = dataset_root
        self.select_idx = select_idx
        self.objs  = readList(os.path.join(self.root, 'objects.txt'), sort=False)
        self.names = readList(os.path.join(self.root, 'names.txt'),   sort=False)
        self.l_dir = light_source_directions()
        self.ints = {}
        self.transform = transform
        ints_name = 'light_intensities.txt'
        for obj in self.objs:
            self.ints[obj] = np.genfromtxt(os.path.join(self.root, obj, ints_name))

    def __getitem__(self, index):
        obj = self.objs[index]
        
        mask = Image.open(os.path.join(self.root, obj, 'mask.png'))
        mask = np.array(mask)
        mask = mask[:,:,0:1]
        

        img_list = [os.path.join(self.root, obj, self.names[i]) for i in self.select_idx]
        dirs = self.l_dir[self.select_idx]

        normal_path = os.path.join(self.root, obj, 'Normal_gt.mat')
        normal = sio.loadmat(normal_path)['Normal_gt']

        norm = np.sqrt((normal * normal).sum(2, keepdims=True))
        normal = normal / (norm + 1e-10)
        normal = np.array(normal)*np.array([-1,1,-1]).reshape(1,1,3)
        sample = {'normal': normal.astype(np.float32),  'mask': mask}

        
        dirs = dirs*np.array([1,-1,1]).reshape(1,3)
        dirs = torch.unbind(torch.from_numpy(dirs).float(),dim=0)

        ints = self.ints[obj][self.select_idx].astype(np.float32)
        sample['ints'] = list(ints)

        sample['name'] = obj
        sample['scene_path'] = os.path.join(self.root, obj)
        
        imgs = []
        for idx, img_name in zip(self.select_idx,img_list):
            #print(img_name)
            #temp = np.array(Image.open(img_name))
            temp =  cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            #print('max',np.max(temp))
            temp = (temp/(2**16)).astype(np.float32)
            if idx < 0:
            	temp = np.zeros(temp.shape).astype(np.float32)
            	
            #print('dtype',temp.dtype)
            imgs.append(temp)
           
        sample['images'] = imgs
        
        if self.transform:
        	self.transform(sample)
        
        return sample

    def __len__(self):
        return len(self.objs)
        


def readList(list_path,ignore_head=False, sort=True):
    lists = []
    with open(list_path) as f:
        lists = f.read().splitlines()
    if ignore_head:
        lists = lists[1:]
    if sort:
        lists.sort(key=natural_keys)
    return lists

def light_source_directions():
    """
    Below matrix is from DiLiGenT.
    :return: light source direction matrix. [light_num, 3]
    :rtype: np.ndarray
    """
    L = np.array([[-0.06059872, -0.44839055, 0.8917812],
                  [-0.05939919, -0.33739538, 0.93948714],
                  [-0.05710194, -0.21230722, 0.97553319],
                  [-0.05360061, -0.07800089, 0.99551134],
                  [-0.04919816, 0.05869781, 0.99706274],
                  [-0.04399823, 0.19019233, 0.98076044],
                  [-0.03839991, 0.31049925, 0.9497977],
                  [-0.03280081, 0.41611025, 0.90872238],
                  [-0.18449839, -0.43989616, 0.87889232],
                  [-0.18870114, -0.32950199, 0.92510557],
                  [-0.1901994, -0.20549935, 0.95999698],
                  [-0.18849605, -0.07269848, 0.97937948],
                  [-0.18329657, 0.06229884, 0.98108166],
                  [-0.17500445, 0.19220488, 0.96562453],
                  [-0.16449474, 0.31129005, 0.93597008],
                  [-0.15270716, 0.4160195, 0.89644202],
                  [-0.30139786, -0.42509698, 0.85349393],
                  [-0.31020115, -0.31660118, 0.89640333],
                  [-0.31489186, -0.19549495, 0.92877599],
                  [-0.31450962, -0.06640203, 0.94692897],
                  [-0.30880699, 0.06470146, 0.94892147],
                  [-0.2981084, 0.19100538, 0.93522635],
                  [-0.28359251, 0.30729189, 0.90837601],
                  [-0.26670649, 0.41020998, 0.87212122],
                  [-0.40709586, -0.40559588, 0.81839168],
                  [-0.41919869, -0.29999906, 0.85689732],
                  [-0.42618633, -0.18329412, 0.88587159],
                  [-0.42691512, -0.05950211, 0.90233197],
                  [-0.42090385, 0.0659006, 0.90470827],
                  [-0.40860354, 0.18720162, 0.89330773],
                  [-0.39141794, 0.29941372, 0.87013988],
                  [-0.3707838, 0.39958255, 0.83836338],
                  [-0.499596, -0.38319693, 0.77689378],
                  [-0.51360334, -0.28130183, 0.81060526],
                  [-0.52190667, -0.16990217, 0.83591069],
                  [-0.52326874, -0.05249686, 0.85054918],
                  [-0.51720021, 0.06620003, 0.85330035],
                  [-0.50428312, 0.18139393, 0.84427174],
                  [-0.48561334, 0.28870793, 0.82512267],
                  [-0.46289771, 0.38549809, 0.79819605],
                  [-0.57853599, -0.35932235, 0.73224555],
                  [-0.59329349, -0.26189713, 0.76119165],
                  [-0.60202327, -0.15630604, 0.78303027],
                  [-0.6037003, -0.04570002, 0.7959004],
                  [-0.59781529, 0.06590169, 0.79892043],
                  [-0.58486953, 0.17439091, 0.79215873],
                  [-0.56588359, 0.27639198, 0.77677747],
                  [-0.54241965, 0.36921337, 0.75462733],
                  [0.05220076, -0.43870637, 0.89711304],
                  [0.05199786, -0.33138635, 0.9420612],
                  [0.05109826, -0.20999284, 0.97636672],
                  [0.04919919, -0.07869871, 0.99568366],
                  [0.04640163, 0.05630197, 0.99733494],
                  [0.04279892, 0.18779527, 0.98127529],
                  [0.03870043, 0.30950341, 0.95011048],
                  [0.03440055, 0.41730662, 0.90811441],
                  [0.17290651, -0.43181626, 0.88523333],
                  [0.17839998, -0.32509996, 0.92869988],
                  [0.18160174, -0.20480196, 0.96180921],
                  [0.18200745, -0.07490306, 0.98044012],
                  [0.17919505, 0.05849838, 0.98207285],
                  [0.17329685, 0.18839658, 0.96668244],
                  [0.1649036, 0.30880674, 0.93672045],
                  [0.1549931, 0.41578148, 0.89616009],
                  [0.28720483, -0.41910705, 0.8613145],
                  [0.29740177, -0.31410186, 0.90160535],
                  [0.30420604, -0.1965039, 0.9321185],
                  [0.30640529, -0.07010121, 0.94931639],
                  [0.30361153, 0.05950226, 0.95093613],
                  [0.29588748, 0.18589214, 0.93696036],
                  [0.28409783, 0.30349768, 0.90949304],
                  [0.26939905, 0.40849857, 0.87209694],
                  [0.39120402, -0.40190413, 0.8279085],
                  [0.40481085, -0.29960803, 0.86392315],
                  [0.41411685, -0.18590756, 0.89103626],
                  [0.41769724, -0.06449957, 0.906294],
                  [0.41498764, 0.05959822, 0.90787296],
                  [0.40607977, 0.18089099, 0.89575537],
                  [0.39179226, 0.29439419, 0.87168279],
                  [0.37379609, 0.39649585, 0.83849122],
                  [0.48278794, -0.38169046, 0.78818031],
                  [0.49848546, -0.28279175, 0.8194761],
                  [0.50918069, -0.1740934, 0.84286803],
                  [0.51360856, -0.05870098, 0.85601427],
                  [0.51097962, 0.05899765, 0.8575658],
                  [0.50151639, 0.17420569, 0.84742769],
                  [0.48600297, 0.28260173, 0.82700506],
                  [0.46600106, 0.38110087, 0.79850181],
                  [0.56150442, -0.35990283, 0.74510586],
                  [0.57807114, -0.26498677, 0.77176147],
                  [0.58933134, -0.1617086, 0.7915421],
                  [0.59407609, -0.05289787, 0.80266769],
                  [0.59157958, 0.057798, 0.80417224],
                  [0.58198189, 0.16649482, 0.79597523],
                  [0.56620006, 0.26940003, 0.77900008],
                  [0.54551481, 0.36380988, 0.7550205]], dtype=float)
    return L


