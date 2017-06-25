import torchvision.transforms as transforms
from options.test_options import TestOptions
from models.test_model import TestModel
from PIL import Image
import argparse

print 'Single Feedforward Test Started...'

# Set options
# python test.py --dataroot ./datasets/maps --name maps_cyclegan --model test --dataset_mode single --phase test
parser = argparse.ArgumentParser()
opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.name = 'maps_cyclegan'
opt.model = 'test'
opt.dataset_mode = 'single'
opt.phase = 'test'

# Create the model
model = TestModel()
model.initialize(opt)

# Data related preparation
transform_list = []
if opt.resize_or_crop == 'resize_and_crop':
    transform_list.append(transforms.Scale(opt.loadSize))

if opt.isTrain and not opt.no_flip:
    transform_list.append(transforms.RandomHorizontalFlip())

if opt.resize_or_crop != 'no_resize':
    transform_list.append(transforms.RandomCrop(opt.fineSize))

transform_list += [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5),
                                        (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

# Load an image
#A_path = "/home/illusion/ML_Linux_SSD_M550/pytorch-CycleGAN-and-pix2pix/datasets/maps/test/736_A.jpg"
A_path = "./datasets/maps/test/1000_A.jpg"
A_img = Image.open(A_path).convert('RGB')
A_img = transform(A_img)
data = {'A': A_img, 'A_paths': A_path}

# Feedforward a test image
model.set_input(data)
model.test()

print 'Test has finished'