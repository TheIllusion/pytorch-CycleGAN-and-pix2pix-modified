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
#opt.batchSize = 1  # test code only supports batchSize = 1
#opt.serial_batches = True  # no shuffle
#opt.no_flip = True  # no flip
#opt.name = 'anonymous'
#opt.model = 'test'
#opt.dataset_mode = 'single'
#opt.phase = 'test'

result_dir = "/home1/irteamsu/users/rklee/gan/pytorch-CycleGAN-and-pix2pix-modified/temp/"
model_path = "/home1/irteamsu/data_ssd/users/rklee/gan/pytorch-CycleGAN-and-pix2pix-modified/checkpoints_for_demo/latest_net_G_A_black_to_blonde.pth"

# Load an image
#A_path = "/home/illusion/ML_Linux_SSD_M550/pytorch-CycleGAN-and-pix2pix/datasets/maps/test/736_A.jpg"
A_path = "/home1/irteamsu/data_ssd/users/rklee/gan/pytorch-CycleGAN-and-pix2pix-modified/temp/result_2017_04_09_22_42_40_187477_97062.jpg"

# Create the model
#model = TestModel()
#model.initialize(opt)

assert (not opt.isTrain)

#input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

gpu_ids = []
gpu_ids.append(0)

tensor = torch.cuda.FloatTensor if gpu_ids else torch.Tensor
input_A = tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

def insert_image(image_full_path):
    
    global input_A
    
    A_img = Image.open(image_full_path).convert('RGB')
    A_img = transform(A_img)
    A_img_reshaped = A_img.view(-1, 3, 256, 256)

    input_A.resize_(A_img_reshaped.size()).copy_(A_img_reshaped)
    
def forward_a_test_image():
    real_A = Variable(input_A)
    fake_B = netG.forward(real_A)

    real_A_im = util.tensor2im(real_A.data)
    fake_B_im = util.tensor2im(fake_B.data)
    visuals = OrderedDict([('real_A', real_A_im), ('fake_B', fake_B_im)])
    return visuals

# Load the generator model
netG = networks.define_G(opt.input_nc, opt.output_nc,
                              opt.ngf, opt.which_model_netG,
                              opt.norm, opt.use_dropout,
                              gpu_ids)

netG.load_state_dict(torch.load(model_path))

print('---------- Networks initialized and loaded -------------')
networks.print_network(netG)
print('--------------------------------------------------------')

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

# Insert a test image
insert_image(A_path)

# Feedforward a test image
visuals = forward_a_test_image()

# Save images
for label, image_numpy in visuals.items():
    image_name = '%s_%s.png' % ("result", label)
    save_path = os.path.join(result_dir, image_name)
    util.save_image(image_numpy, save_path)

print 'Test has finished'