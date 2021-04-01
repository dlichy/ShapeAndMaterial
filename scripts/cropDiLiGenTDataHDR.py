import os, argparse, sys, shutil, glob
import numpy as np
import re
from PIL import Image
import scipy.io as sio
import cv2
root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(root_path)

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',  default='data/DiLiGenT/pmsData')
parser.add_argument('--obj_list',   default='objects.txt')
parser.add_argument('--suffix',     default='crop')
parser.add_argument('--file_ext',   default='.png')
parser.add_argument('--normal_name',default='Normal_gt.png')
parser.add_argument('--mask_name',  default='mask.png')
parser.add_argument('--n_key',      default='Normal_gt')
parser.add_argument('--pad',        default=15, type=int)
args = parser.parse_args()

def readList(list_path,ignore_head=False, sort=True):
    lists = []
    with open(list_path) as f:
        lists = f.read().splitlines()
    if ignore_head:
        lists = lists[1:]
    if sort:
        lists.sort(key=natural_keys)
    return lists
    
def makeFile(f):
    if not os.path.exists(f):
        os.makedirs(f)
    #else:  raise Exception('Rendered image directory %s is already existed!!!' % directory)

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]
    
def atoi(text):
    return int(text) if text.isdigit() else text

def getSaveDir():
    dirName  = os.path.dirname(args.input_dir)
    save_dir = '%s_%s' % (args.input_dir, args.suffix) 
    makeFile(save_dir)
    print('Output dir: %s\n' % save_dir)
    return save_dir

def getBBoxCompact(mask):
    index = np.where(mask != 0)
    t, b, l , r = index[0].min(), index[0].max(), index[1].min(), index[1].max()
    h, w = b - t + 2 * args.pad, r - l + 2 * args.pad
    t = max(0, t - args.pad)
    b = t + h 
    l = max(0, l - args.pad)
    r = l + w 
    if h % 4 != 0: 
        pad = 4 - h % 4
        b += pad; h += pad
    if w % 4 != 0: 
        pad = 4 - w % 4
        r += pad; w += pad
    return l, r, t, b, h, w

def loadMaskNormal(d):
    mask   = np.array(Image.open(os.path.join(args.input_dir, d, args.mask_name)))
    try:
        normal = np.array(Image.open(os.path.join(args.input_dir, d, args.normal_name)))
    except IOError:
        normal = np.array(Image.open(os.path.join(args.input_dir, d, 'normal_gt.png')))
    n_mat  = sio.loadmat(os.path.join(args.input_dir, d, 'Normal_gt.mat'))[args.n_key]
    h, w, c = normal.shape
    print('Processing Objects: %s' % d, mask.shape)
    if mask.ndim < 3:
        mask = mask.reshape(h, w, 1).repeat(3, 2)
    return mask, normal, n_mat

def copyTXT(d):
    txt = glob.glob(os.path.join(args.input_dir, '*.txt'))
    for t in txt:
        name = os.path.basename(t)
        shutil.copy(t, os.path.join(args.save_dir, name))

    txt = glob.glob(os.path.join(args.input_dir, d, '*.txt'))
    for t in txt:
        name = os.path.basename(t)
        shutil.copy(t, os.path.join(args.save_dir, d, name))

if __name__ == '__main__':
    print('Input dir: %s\n' % args.input_dir)
    args.save_dir = getSaveDir()

    dir_list  = readList(os.path.join(args.input_dir, args.obj_list))
    name_list = readList(os.path.join(args.input_dir, 'names.txt'))
    max_h, max_w = 0, 0
    crop_list = open(os.path.join(args.save_dir, 'crop.txt'), 'w')
    for d in dir_list:
        makeFile(os.path.join(args.save_dir, d))
        mask, normal, n_mat = loadMaskNormal(d)
        l, r, t, b, h, w = getBBoxCompact(mask[:,:,0] / 255)
        crop_list.write('%d %d %d %d %d %d\n' % (mask.shape[0], mask.shape[1], l, r, t, b))

        max_h = h if h > max_h else max_h
        max_w = w if w > max_w else max_w
        print('\t BBox L %d R %d T %d B %d, H:%d W:%d, Padded: %d %d' % 
                (l, r, t, b, h, w, r - l, b - t)) 
        imsave(os.path.join(args.save_dir, d, args.mask_name), mask[t:b, l:r, :])
        imsave(os.path.join(args.save_dir, d, args.normal_name), normal[t:b, l:r, :])
        sio.savemat(os.path.join(args.save_dir, d, 'Normal_gt.mat'), 
                {args.n_key: n_mat[t:b, l:r, :]} ,do_compression=True)
        copyTXT(d)
        intens = np.genfromtxt(os.path.join(args.input_dir, d, 'light_intensities.txt'))
        for idx, name in enumerate(name_list):
            img = cv2.imread(os.path.join(args.input_dir, d, name), cv2.IMREAD_UNCHANGED)
            #img = img.astype(np.float32)/(2**16)
            img = img[t:b, l:r, :]
            cv2.imwrite(os.path.join(args.save_dir, d, name), img)
    print('Max H %d, Max %d' % (max_h, max_w))
    crop_list.close()
