from .sample_transforms import *

all_domains = ['albedo','normal','rough','mask']

size_transform = RandomChooseTransform([ShrinkAndPad(keys=all_domains,list_keys=['images'],my_range=(0.6,1)),
                                         RandomCropResize((256,256),keys=all_domains,list_keys=['images'], scale=(0.7,1))]
                                         ,[0.3,0.7])
                                         
train_synthetic_transforms = transforms.Compose([
      NormalizeByMedian(list_keys=['images']),
      size_transform,
      RandomScaleUnique(list_keys=['images'],my_range=(0.01,0.2)),
      TonemapSRGB(list_keys=['images']),
      ErodeMask(),
      RemoveNans(keys=all_domains,list_keys=['images']),
      MyToTensor(keys=all_domains,list_keys=['images'])
      ])


test_synthetic_transforms = transforms.Compose([
      NormalizeByMedian(list_keys=['images']),
      RandomScaleUnique(list_keys=['images'],my_range=(0.1,0.2)),
      TonemapSRGB(list_keys=['images']),
      ErodeMask(),
      RemoveNans(keys=all_domains,list_keys=['images']),
      MyToTensor(keys=all_domains,list_keys=['images'])
      ])
      
      
diligent_transforms = transforms.Compose([
	TonemapSRGB(list_keys=['images']),
	PadToSquare(keys=['mask','normal'],list_keys=['images'],record_pad_shape=True),
	PadSquareToPower2(keys=['mask','normal'],list_keys=['images'],record_pad_shape=True),
	MyToTensor(keys=['mask','normal'],list_keys=['images'])])
	
	
standard_srgb_transforms = transforms.Compose([
	PadToSquare(keys=['mask'],list_keys=['images'],record_pad_shape=True),
	PadSquareToPower2(keys=['mask'],list_keys=['images'],record_pad_shape=True),
	MyToTensor(keys=['mask'],list_keys=['images'])])
