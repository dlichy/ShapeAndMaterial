import torch.optim as optim
from models.criterion import *
from models.recnet import *
from models.save_load_checkpoint import *
from datasets.real_dataset import *
from datasets.dataset_utils.common_transforms import *
from run_epoch import run_epoch
from utils.logger import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('output_dir', default=None, help='the path to outputs')
parser.add_argument('--gpu', default=False,  action='store_true', help='enable to run on gpu')
parser.add_argument('--save_all_scales', default=False,  action='store_true', help='save outputs at all scales, if not enabled save only the largest scale')
parser.add_argument('--dataset_root', default='data/sample_dataset', help='path to dataset')
parser.add_argument('--to_zero', type=int, nargs="+", default=[], help='images to zero')
parser.add_argument('--save_ext', default='.png', choices=['.png','.exr'], help='format to save network outputs. If using .exr must have OpenEXR installed')
parser.add_argument('--checkpoint', default='pretrained_models/cvpr_2021_improved.pth', help='path to checkpoint to load')
opt = parser.parse_args()


dataset = RealDataset(opt.dataset_root, to_zero=opt.to_zero, transform = standard_srgb_transforms)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

net = RecNet()
if opt.gpu:
	device = 'cuda'
	net.to(device)
else:
	if torch.cuda.is_available():
		import warnings
		warnings.warn('running on CPU but GPUs detected. Add arg "--gpu" to run on gpu')
	device='cpu'
	
load_checkpoint(opt.checkpoint, net=net,device=device)


criterion = Criterion()


#scalar_logger = ScalarLogger(os.path.join(opt.output_dir,'test_diligent.txt'), log_freq=1, keys=['loss','angular_error'])
image_logger = ImageLogger(os.path.join(opt.output_dir,'images'),log_freq=1,indices_to_save=[0],save_csv_for_integration=True,save_last_only=not opt.save_all_scales, save_ext=opt.save_ext)

with torch.no_grad():
	run_epoch(net, False, dataloader, device,  criterion=criterion, scalar_logger=None, image_logger=image_logger)
#scalar_logger.summarize()
image_logger.write_csv()
