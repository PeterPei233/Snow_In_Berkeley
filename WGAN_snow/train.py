import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim

import functools
import numpy as np
import time
from PIL import Image
import os

from torch.utils.data import Dataset
import torchvision
import os
from PIL import Image

def filename_filter(filename):
    if filename.startswith('.'):
        return False
    lower = filename.lower()
    if lower.endswith('.jpg') or lower.endswith('.png') \
        or lower.endswith('.jpeg')  or lower.endswith('.bmp'):
        return True
    return False
    
class ImageFolder(Dataset):
    def __init__(self, root_dir, transform=None, lazy=True):
        lst = [f for f in os.listdir(root_dir) 
                 if filename_filter(f)]

        self.lst = lst
        self.root_dir = root_dir
        self.transform = transform
        self.lazy = lazy
        
        if not lazy:
            from tqdm.auto import tqdm
            self.imgs = []
            for filename in tqdm(lst):
                self.imgs.append(self._read_image(filename))
            assert len(self.imgs) == len(self.lst)
    
    def __len__(self):
        return len(self.lst)
    
    def __getitem__(self, idx):
        if not self.lazy:
            image = self.imgs[idx]
        else:
            image = self._read_image(self.lst[idx])
            
        if self.transform is not None:
            image = self.transform(image)
        return image
    
    def _read_image(self, filename):
        img_name = os.path.join(self.root_dir, filename)
        return Image.open(img_name).convert('RGB')
    
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
    
class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True, use_bias=True, n_blocks=9, padding_type='reflect', add_method='add'):
        super().__init__()
        
        if use_bias is None:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.main = nn.Sequential(*model)

        if add_method == 'add':
            self.add = torch.add
        else:
            raise ValueError

    def forward(self, input):
        output = self.main(input)
        output = self.add(output, input)
        output = output.clamp(-1., 1.)
        return output
    
class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    
    
import os, sys
# sys.path.append(os.getcwd())

import time
import functools
import argparse

import numpy as np
#import sklearn.datasets

# import libs as lib
# import libs.plot
# from tensorboardX import SummaryWriter

# from models.wgan import *

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import grad
from timeit import default_timer as timer

import torch.nn.init as init

DATA_DIR = 'sidewalk'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_64x64.py!')

RESTORE_MODE = False # if True, it will load saved model from OUT_PATH and continue to train

# MODE = 'wgan-gp'
OUTPUT_PATH = 'snowTue1038/'

continue_from = 749
START_ITER = 750 # starting iteration 
END_ITER = 2000 # How many iterations to train for

os.makedirs(OUTPUT_PATH, exist_ok=True)

CRITIC_ITERS = 5 # How many iterations to train the critic for
GENER_ITERS = 1
N_GPUS = 1 # Number of GPUs
BATCH_SIZE = 32 # Batch size. Must be a multiple of N_GPUS
LAMBDA = 10 # Gradient penalty lambda hyperparameter
IMGSIZE = 256
CHANNELS = 3
OUTPUT_DIM = IMGSIZE*IMGSIZE*CHANNELS # Number of pixels in each image
NUM_WORKERS = 6

# def showMemoryUsage(device=1):
#     gpu_stats = gpustat.GPUStatCollection.new_query()
#     item = gpu_stats.jsonify()["gpus"][device]
#     print('Used/total: ' + "{}/{}".format(item["memory.used"], item["memory.total"]))

def load_data(path_to_folder, classes=None):
    preprocess = torchvision.transforms.Compose([
               torchvision.transforms.RandomRotation(degrees=3, resample=Image.BICUBIC),
               torchvision.transforms.RandomResizedCrop(IMGSIZE+IMGSIZE//20, (0.25, 1.0), interpolation=Image.BICUBIC),
               torchvision.transforms.CenterCrop(IMGSIZE),
               torchvision.transforms.RandomHorizontalFlip(p=0.5),
               torchvision.transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
               torchvision.transforms.ToTensor(),
               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
           ])
    dataset = ImageFolder(path_to_folder, transform=preprocess, lazy=False)
    print(len(dataset))
    dataset_loader = torch.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
    return dataset_loader

def training_data_loader(subdir):
    return load_data(os.path.join(DATA_DIR, subdir)) 

# def val_data_loader():
#     return load_data(VAL_DIR, VAL_CLASS) 

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous()
    alpha = alpha.view(BATCH_SIZE, CHANNELS, IMGSIZE, IMGSIZE)
    alpha = alpha.to(device)
    
    fake_data = fake_data.view(BATCH_SIZE, CHANNELS, IMGSIZE, IMGSIZE)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def generate_image(netG, input):
    with torch.no_grad():
        samples = netG(input)
        samples = samples.view(BATCH_SIZE, CHANNELS, IMGSIZE, IMGSIZE)
        samples = samples * 0.5 + 0.5
    return samples
# def generate_image(netG, noise=None):
#     if noise is None:
#         noise = gen_rand_noise()

#     with torch.no_grad():
#     	noisev = noise 
#     samples = netG(noisev)
#     samples = samples.view(BATCH_SIZE, CHANNELS, IMGSIZE, IMGSIZE)
#     samples = samples * 0.5 + 0.5
#     return samples

def gen_rand_noise():
    raise NotImplementedError
#     noise = torch.randn(BATCH_SIZE, 128)
#     noise = noise.to(device)

#     return noise

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
# fixed_noise = gen_rand_noise() 

trainA = training_data_loader('trainA')
trainB = training_data_loader('trainB')
print(f"Train A: {len(trainA)}, train B: {len(trainB)}")

iterA = iter(trainA)
fixed_input = next(iterA).to(device)

if continue_from is not None:
    aG = torch.load(OUTPUT_PATH + f"generator_{continue_from}.pt")
    aD = torch.load(OUTPUT_PATH + f"discriminator_{continue_from}.pt")
else:
    aG = Generator()
    aD = Discriminator()
    
#     aG.apply(weights_init)
#     aD.apply(weights_init)

LR = 2e-4
optimizer_g = torch.optim.Adam(aG.parameters(), lr=LR, betas=(0.5,0.999))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=LR, betas=(0.5,0.999))
one = torch.FloatTensor([1])
mone = one * -1
aG = aG.to(device)
aD = aD.to(device)
one = one.to(device)
mone = mone.to(device)

print(fixed_input.is_cuda)
print(fixed_input.requires_grad)

from tensorboardX import SummaryWriter
writer = SummaryWriter(OUTPUT_PATH + 'logs/')
torchvision.utils.save_image(fixed_input * 0.5 + 0.5, OUTPUT_PATH + 'source.png', nrow=4, padding=2)
grid_images = torchvision.utils.make_grid(fixed_input * 0.5 + 0.5, nrow=4, padding=2)
writer.add_image('images', grid_images, 0)


#Reference: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
def train():
    iterA = iter(trainA)
    iterB = iter(trainB)
    for iteration in range(START_ITER, END_ITER):
        start_time = time.time()
#         print("Iter: " + str(iteration))
#         start = timer()
        #---------------------TRAIN G------------------------
        for p in aD.parameters():
            p.requires_grad_(False)  # freeze D

        gen_cost = None
        for i in range(GENER_ITERS):
#             print("Generator iters: " + str(i))
            aG.zero_grad()
            input = next(iterA, None)
            if input is None:
                iterA = iter(trainA)
                input = next(iterA)
#             noise = gen_rand_noise()
#             noise.requires_grad_(True)
#             fake_data = aG(noise)
            input = input.to(device)
            fake_data = aG(input)
            gen_cost = aD(fake_data)
            gen_cost = gen_cost.mean()
            gen_cost.backward(mone)
            gen_cost = -gen_cost
        
        optimizer_g.step()
#         end = timer()
#         print(f'---train G elapsed time: {end - start}')
        #---------------------TRAIN D------------------------
        for p in aD.parameters():  # reset requires_grad
            p.requires_grad_(True)  # they are set to False below in training G
        for i in range(CRITIC_ITERS):
#             print("Critic iter: " + str(i))
            
#             start = timer()
            aD.zero_grad()

            # gen fake data and load real data
#             noise = gen_rand_noise()
            input = next(iterA, None)
            if input is None:
                iterA = iter(trainA)
                input = next(iterA)
            input = input.to(device)
            with torch.no_grad():
                fake_data = aG(input)
#                 noisev = noise  # totally freeze G, training D
#                 fake_data = aG(noisev).detach()
#             end = timer(); print(f'---gen G elapsed time: {end-start:.2}')
#             print(fake_data.requires_grad) # returns False
#             start = timer()
            batch = next(iterB, None)
            if batch is None:
                iterB = iter(trainB)
                batch = iterB.next()
#             batch = batch[0] #batch[1] contains labels
#             print(batch.mean())
            real_data = batch.to(device) #TODO: modify load_data for each loading
#             end = timer(); print(f'---load real imgs elapsed time: {end-start:.2}')
#             start = timer()

            # train with real data
            disc_real = aD(real_data)
            disc_real = disc_real.mean()

            # train with fake data
            disc_fake = aD(fake_data)
            disc_fake = disc_fake.mean()

            #showMemoryUsage(0)
            # train with interpolates data
            gradient_penalty = calc_gradient_penalty(aD, real_data, fake_data)
            #showMemoryUsage(0)

            # final disc cost
            disc_cost = disc_fake - disc_real + gradient_penalty
            disc_cost.backward()
            w_dist = disc_fake  - disc_real
            optimizer_d.step()
            #------------------VISUALIZATION----------
            if i == CRITIC_ITERS-1:
                print(f"iter {iteration} | disc_cost: {disc_cost:.4} | grad_pen: {gradient_penalty:.4}")
                writer.add_scalar('data/disc_cost', disc_cost, iteration)
                #writer.add_scalar('data/disc_fake', disc_fake, iteration)
                #writer.add_scalar('data/disc_real', disc_real, iteration)
                writer.add_scalar('data/gradient_pen', gradient_penalty, iteration)
                #writer.add_scalar('data/d_conv_weight_mean', [i for i in aD.children()][0].conv.weight.data.clone().mean(), iteration)
                #writer.add_scalar('data/d_linear_weight_mean', [i for i in aD.children()][-1].weight.data.clone().mean(), iteration)
                #writer.add_scalar('data/fake_data_mean', fake_data.mean())
                #writer.add_scalar('data/real_data_mean', real_data.mean())
                #if iteration %200==99:
                #    paramsD = aD.named_parameters()
                #    for name, pD in paramsD:
                #        writer.add_histogram("D." + name, pD.clone().data.cpu().numpy(), iteration)
#                 if iteration %200==199:
#                     body_model = [i for i in aD.children()][0]
#                     layer1 = body_model.conv
#                     xyz = layer1.weight.data.clone()
#                     tensor = xyz.cpu()
#                     tensors = torchvision.utils.make_grid(tensor, nrow=8,padding=1)
#                     writer.add_image('D/conv1', tensors, iteration)

#             end = timer(); print(f'---train D elapsed time: {end-start:.2}')
        #---------------VISUALIZATION---------------------
        writer.add_scalar('data/gen_cost', gen_cost, iteration)
        print(f"iter {iteration} | gen_cost: {gen_cost:.4} | time: {time.time() - start_time:.2f}s")
#         lib.plot.plot(OUTPUT_PATH + 'time', time.time() - start_time)
#         lib.plot.plot(OUTPUT_PATH + 'train_disc_cost', disc_cost.cpu().data.numpy())
#         lib.plot.plot(OUTPUT_PATH + 'train_gen_cost', gen_cost.cpu().data.numpy())
#         lib.plot.plot(OUTPUT_PATH + 'wasserstein_distance', w_dist.cpu().data.numpy())
        if iteration % 10 == 9:
#             val_loader = val_data_loader() 
#             dev_disc_costs = []
#             for _, images in enumerate(val_loader):
#                 imgs = torch.Tensor(images[0])
#                	imgs = imgs.to(device)
#                 with torch.no_grad():
#             	    imgs_v = imgs

#                 D = aD(imgs_v)
#                 _dev_disc_cost = -D.mean().cpu().data.numpy()
#                 dev_disc_costs.append(_dev_disc_cost)
#             lib.plot.plot(OUTPUT_PATH + 'dev_disc_cost.png', np.mean(dev_disc_costs))
#             lib.plot.flush()	
#             showMemoryUsage(0)
            gen_images = generate_image(aG, fixed_input)
            torchvision.utils.save_image(gen_images, OUTPUT_PATH + 'samples_{}.png'.format(iteration), nrow=4, padding=2)
            grid_images = torchvision.utils.make_grid(gen_images, nrow=4, padding=2)
            writer.add_image('images', grid_images, iteration)
	#----------------------Save model----------------------
        if iteration % 25 == 24:
            torch.save(aG, OUTPUT_PATH + f"generator_{iteration}.pt")
            torch.save(aD, OUTPUT_PATH + f"discriminator_{iteration}.pt")
#         lib.plot.tick()

train()