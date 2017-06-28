import torchvision.transforms as transforms
import argparse
import os
import torch
from collections import OrderedDict
from options.test_options import TestOptions
from models.test_model import TestModel
from PIL import Image
from util.visualizer import Visualizer
from util import util
from models import networks
from torch.autograd import Variable

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

result_dir = "/media/illusion/ML_DATA_SSD_M550/pytorch-CycleGAN-and-pix2pix/temp/"
model_path = "/media/illusion/ML_DATA_SSD_M550/pytorch-CycleGAN-and-pix2pix/checkpoints/maps_cyclegan/1_net_G_A.pth"

# Create the model
model = TestModel()
model.initialize(opt)

assert (not opt.isTrain)

#input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

gpu_ids = []
gpu_ids.append(0)

tensor = torch.cuda.FloatTensor if gpu_ids else torch.Tensor
input_A = tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

netG = networks.define_G(opt.input_nc, opt.output_nc,
                              opt.ngf, opt.which_model_netG,
                              opt.norm, opt.use_dropout,
                              gpu_ids)

print('---------- Networks initialized -------------')
networks.print_network(netG)
print('-----------------------------------------------')

netG.load_state_dict(torch.load(model_path))

visualizer = Visualizer(opt)

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
A_img_reshaped = A_img.view(-1, 3, 256, 256)

data = {'A': A_img_reshaped, 'A_paths': A_path}

# we need to use single_dataset mode
input_A_temp = data['A']
input_A.resize_(input_A_temp.size()).copy_(input_A_temp)
image_paths = data['A_paths']

# Feedforward a test image
real_A = Variable(input_A)
fake_B = netG.forward(real_A)

#visuals = model.get_current_visuals()
real_A_im = util.tensor2im(real_A.data)
fake_B_im = util.tensor2im(fake_B.data)
visuals = OrderedDict([('real_A', real_A_im), ('fake_B', fake_B_im)])

for label, image_numpy in visuals.items():
    image_name = '%s_%s.png' % ("result", label)
    save_path = os.path.join(result_dir, image_name)
    util.save_image(image_numpy, save_path)

print 'Test has finished'