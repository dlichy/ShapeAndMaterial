
**[Shape and Material Capture at Home, CVPR 2021]**.
<br>
Daniel Lichy, Jiaye Wu, [Soumyadip Sengupta](https://homes.cs.washington.edu/~soumya91/), [David Jacobs](http://www.cs.umd.edu/~djacobs/)
<br>

This is the official code release for the paper *Shape and Material Capture at Home*. The code enables you to
reconstruct a 3D mesh and Cook-Torrance BRDF from one or more images captured with a flashlight or camera flash. 

## Dependencies
This project uses the following dependencies:

- Python 3.8
- PyTorch (version = 1.8.1)
- torchvision
- numpy
- scipy
- opencv
- OpenEXR (only required for training)

The easiest way to run the code is by creating a virtual environment and installing the dependences with pip e.g.
```shell
# Create a new python3.8 environment named py3.8
virtualenv py3.8 python=3.8

# Activate the created environment
source py3.8

# To install dependencies 
python3 -m pip install -r requirements.txt
#or
python3 -m pip install -r requirements_no_exr.txt


```
## Overview
We provide:
- The trained RecNet model.
- Code to test on the DiLiGenT dataset.
- Code to test on our dataset from the paper.
- Code to test on your own dataset.
- Code to train a new model, including code for visualization and logging.

## Capturing you own dataset
### Multi-image captures
The video below shows how to capture the (up to) six images for you own dataset. Angles are approximate and can be estimated by eye.

### Single image captures
Our network also provides state-of-the-art results for reconstructing shape and material from a single flash image.

Examples captured with just an iPhone with flash enabled in a dim room (complete darkness is not needed):

Coming soon...

### Mask Making
For best performance you should supply a segmentation mask with you image. For our paper we used https://github.com/saic-vul/fbrs_interactive_segmentation
which enables mask making with just a few clicks.

Normals results are reasonable without the mask, but integrating normals to a mesh can be challenging.

## Test RecNet on the DiLiGenT dataset
```shell
# Download and prepare the DiLiGenT dataset
sh scripts/prepare_diligent_dataset.sh

# Test on 3 DiLiGenT images from the front, front-right, and front-left
# if you only have CPUs remove the --gpu argument
python eval_diligent.py results_path --gpu

# To test on a different subset of DiLiGenT images use the argument --image_nums n1 n2 n3 n4 n5 n6
# where n1 to n6 are the image indices of the right, front-right, front, front-left, left, and above
# images, respectively. For images that are no present set the image number to -1
# e.g to test on only the front image (image number 51) run
python eval_diligent.py results_path --gpu --image_nums -1 -1 51 -1 -1 -1 
```

## Test on our dataset/your own dataset
The easiest way to test on you own dataset and our dataset is to format it as follows:

dataset_dir:
* sample_name1:
  * 0.ext (right)
  * 1.ext (front-right)
  * 2.ext (front)
  * 3.ext (front-left)
  * 4.ext (left)
  * 5.ext (above)
  * mask.ext
* sample_name2: (if not all images are present just don't add it to the directory)
  * 2.ext (front)
  * 3.ext (front-left)
* ...

Where *.ext* is the image extention e.g. .png, .jpg, .exr

For an example of formating your own dataset please look in data/sample_dataset

Then run:
```shell
python eval_standard.py results_path --dataset_root path_to_dataset_dir --gpu

# To test on a sample of our dataset run
python eval_standard.py results_path --dataset_root data/sample_dataset --gpu
```

### Download our real dataset
Coming Soon...

## Integrating Normal Maps and Producing a Mesh
We include a script to integrate normals and produce a ply mesh with per vertex albedo and roughness.

After running eval_standard.py or eval_diligent.py there with be a file results_path/images/integration_data.csv
Running the following command with produce a ply mesh in results_path/images/sample_name/mesh.ply
```shell
python integrate_normals.py results_path/images/integration_data.csv --gpu
```

## Training
To train RecNet from scratch:
```shell
python train.py log_dir --dr_dataset_root path_to_dr_dataset --sculpt_dataset_root path_to_sculpture_dataset --gpu
```
### Download the training data
Coming Soon...


## FAQ
#### Q1: What should I do if I have problem in running your code?
- Please create an issue if you encounter errors when trying to run the code. Please also feel free to submit a bug report.


## Citation
If you find this code or the provided models useful in your research, please cite it as: 
```
@inproceedings{lichy_2021,
  title={Shape and Material Capture at Home},
  author={Lichy, Daniel and Wu, Jiaye and Sengupta, Soumyadip and Jacobs, David W.},
  booktitle={CVPR},
  year={2021}
}
```

### Acknowledgement 
Code used for downloading and loading the DiLiGenT dataset is adapted from https://github.com/guanyingc/SDPS-Net
