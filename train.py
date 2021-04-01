import torch.optim as optim
from models.criterion import *
from models.recnet import *
from datasets.synthetic_dataset import *
from models.save_load_checkpoint import *
from datasets.dataset_utils.common_transforms import *
from run_epoch import run_epoch
from utils.logger import *
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('logdir', default=None, help='the path to store logging information and models and models')
parser.add_argument('--gpu', default=False,  action='store_true', help='enable to run on gpu')
# The location of training set
parser.add_argument('--dr_dataset_root', help='path to random object dataset')
parser.add_argument('--sculpture_dataset_root', help='path to sculpture dataset')
# The basic training setting
parser.add_argument('--epochs', type=int, default=50, help='the number of epochs for training')
parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
# The training weight
parser.add_argument('--albedo_weight', type=float, default=1.0, help='the weight for the diffuse component')
parser.add_argument('--normal_weight', type=float, default=1.0, help='the weight for the diffuse component')
parser.add_argument('--rough_weight', type=float, default=5.0, help='the weight for the roughness component')
parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
#logging options
parser.add_argument('--scalars_to_log', type=list, default=['loss','albedo_loss','normal_loss','rough_loss','angular_error'], help='the scalars to log')
parser.add_argument('--test_scalar_lf', type=int, default=20, help='frequency to log scalars during testing')
parser.add_argument('--train_scalar_lf', type=int, default=50, help='frequency to log scalars during training')
parser.add_argument('--test_image_lf', type=int, default=20, help='frequency to log images during testing')
parser.add_argument('--train_image_lf', type=int, default=200, help='frequency to log images during testing')
parser.add_argument('--checkpoint_freq', type=int, default=2, help='how frequently to save model weights')
parser.add_argument('--save_last_only', default=False,  action='store_true', help='only save outputs at largest scale')
#resume training from checkpoint
parser.add_argument('--checkpoint', default='None', help='path to checkpoint to load')


opt = parser.parse_args()
if opt.gpu:
	device = 'cuda'
else:
	if torch.cuda.is_available():
		import warnings
		warnings.warn('running on CPU but GPUs detected. Add arg \"--gpu\" to run on gpu')
	device='cpu'

train_datasets = {}
test_datasets = {}

if not (opt.dr_dataset_root or opt.sculpture_dataset_root):
	raise Exception('must specify a training at least one training dataset root')

if opt.dr_dataset_root:
	train_datasets['dr_train_data'] = SyntheticDataset(opt.dr_dataset_root, True, colocated_as_2=True, transform=train_synthetic_transforms, mode='random')
	test_datasets['dr_test_data_1'] = SyntheticDataset(opt.dr_dataset_root, False, colocated_as_2=True, transform=test_synthetic_transforms, mode='select',selected_lights=[2])
	test_datasets['dr_test_data_3'] = SyntheticDataset(opt.dr_dataset_root, False, colocated_as_2=True, transform=test_synthetic_transforms, mode='select',selected_lights=[1,2,3])
	test_datasets['dr_test_data_6'] = SyntheticDataset(opt.dr_dataset_root, False, colocated_as_2=True, transform=test_synthetic_transforms,  mode='select')
	
if opt.sculpture_dataset_root:
	train_datasets['sculpt_train_data'] = SyntheticDataset(opt.sculpture_dataset_root, True, colocated_as_2=True, transform=train_synthetic_transforms, mode='random')
	test_datasets['sculpt_test_data'] = SyntheticDataset(opt.sculpture_dataset_root, False, colocated_as_2=True, transform=test_synthetic_transforms, mode='random')
	

#concatenate train datasets
train_data = torch.utils.data.ConcatDataset(train_datasets.values())
train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=8)

#test datasets are kept separate
test_loaders = {}
for k,v in test_datasets.items():
	test_loaders[k] = torch.utils.data.DataLoader(v, batch_size=opt.batch_size, shuffle=False, num_workers=8)


net=RecNet()
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=opt.lr)

if opt.checkpoint == 'None':
	start_epoch = 0
else:
	start_epoch = load_checkpoint(opt.checkpoint, net=net, optimizer=optimizer)

if opt.gpu:
	net = nn.DataParallel(net)	

criterion = Criterion(albedo_weight=opt.albedo_weight,normal_weight=opt.normal_weight,rough_weight=opt.rough_weight)


#make logdir
if not os.path.exists(opt.logdir):
	os.mkdir(opt.logdir)


for epoch in range(start_epoch,opt.epochs):
	epoch_dir = os.path.join(opt.logdir,'epoch_{}'.format(epoch))
	if not os.path.exists(epoch_dir):
		os.mkdir(epoch_dir)
	
	#make train logs
	train_image_dir = os.path.join(epoch_dir,'train_images')
	if not os.path.exists(train_image_dir):
		os.mkdir(train_image_dir)
	
	#train
	scalar_logger = ScalarLogger(os.path.join(epoch_dir,'train_log.txt'), log_freq=opt.train_scalar_lf, keys=opt.scalars_to_log)
	image_logger = ImageLogger(train_image_dir,log_freq=opt.train_image_lf,indices_to_save=[0],save_last_only=opt.save_last_only)
	run_epoch(net, True, train_loader, device,  criterion=criterion, optimizer=optimizer, scalar_logger=scalar_logger, image_logger=image_logger)
	scalar_logger.summarize()
	
	
	#test with all test sets
	for k,v in test_loaders.items():
		test_image_dir = os.path.join(epoch_dir,'test_images_{}'.format(k))
		if not os.path.exists(test_image_dir):
			os.mkdir(test_image_dir)

		#test
		scalar_logger = ScalarLogger(os.path.join(epoch_dir,'eval_log_{}.txt'.format(k)), log_freq=opt.test_scalar_lf, keys=opt.scalars_to_log)
		image_logger = ImageLogger(test_image_dir,log_freq=opt.test_image_lf,indices_to_save=[0])
		with torch.no_grad():
			run_epoch(net, False, v, device,  criterion=criterion, scalar_logger=scalar_logger, image_logger=image_logger)
		scalar_logger.summarize()
	
	#checkpoint
	if epoch % opt.checkpoint_freq == 0:
		save_checkpoint(os.path.join(epoch_dir,'checkpoint_{}.pth'.format(epoch)), net=net, optimizer=optimizer)
	
	
	
