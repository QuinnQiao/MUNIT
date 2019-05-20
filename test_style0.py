from utils import get_config, get_data_loader_folder, get_data_loader_folder_centercrop, pytorch03_to_pytorch04
from trainer import MUNIT_Trainer
from torch import nn
import torch.nn.functional as F
import argparse
import numpy as np
import torchvision.utils as vutils
import sys
import torch
import os
from PIL import Image
import torchvision.transforms as transforms


def save_images(save_dir, save_name, images, num_input):
    image_tensor = torch.cat(images, 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=num_input, padding=0, normalize=True)
    vutils.save_image(image_grid, os.path.join(save_dir, save_name), nrow=1)


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder', help='Path to the config file.')
parser.add_argument('--input_folder', type=str, help="input image folder")
parser.add_argument('--output_folder', type=str, help="output image folder")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--style_interpolate',type=int, default=5, help="number of styles to sample")
parser.add_argument('--style_folder', type=str, default=None, help="style image folder")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--gpu_id', type=int, default=0, help="which gpu to use")
parser.add_argument('--centercrop', action='store_true', default=False, help='if centercrop when load dataset')

opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

device = 'cuda:%d'%opts.gpu_id

config = get_config(opts.config)

config['vgg_model_path'] = opts.output_path

if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

style_dim = config['gen']['style_dim']
trainer = MUNIT_Trainer(config, device)

try:
    state_dict = torch.load(opts.checkpoint)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
except:
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), 'MUNIT')
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

trainer.to(device)
trainer.eval()
encode1, encode2 = trainer.gen_a.encode, trainer.gen_b.encode # encode function
decode1, decode2 = trainer.gen_a.decode, trainer.gen_b.decode # decode function


get_loader = get_data_loader_folder_centercrop if opts.centercrop else get_data_loader_folder

content_loader = get_loader(opts.input_folder, 1, False, new_size=config['new_size'], 
                            crop=True, height=config['new_size'], width=config['new_size'])
style_loader = get_loader(opts.style_folder, 1, False, new_size=config['new_size'], 
                            crop=True, height=config['new_size'], width=config['new_size'])

content1, style1 = [], []
num_input1 = 0
for data in content_loader:
    num_input1 += 1
    data = data.to(device)
    with torch.no_grad():
        content, style = encode1(data)
    content1.append(content)
    style1.append(style)

content2, style2 = [], []
num_input2 = 0
for data in style_loader:
    num_input2 += 1
    data = data.to(device)
    with torch.no_grad():
        content, style = encode2(data)
        content2.append(content)
        style2.append(style)

num_input = min(num_input1, num_input2)
content1 = content1[:num_input]
content2 = content2[:num_input]
style1 = style1[:num_input]
style2 = style2[:num_input]

style1_random = torch.randn(*style1[0].size(), device=style1[0].device)
step1 = style1_random / opts.style_interpolate
# within-domain content
image1 = []
for i in range(num_input):
    with torch.no_grad():
        img = decode1(content1[i], style1[i])
        image1.append((img.cpu().data+1)/2)
        for j in range(2*opts.style_interpolate+1):
            style = style1_random - step1*j
            img = decode1(content1[i], style)
            image1.append((img.cpu().data+1)/2)
# cross-domain content
for i in range(num_input):
    with torch.no_grad():
        img = decode1(content2[i], style1[i])
        image1.append((img.cpu().data+1)/2)
        for j in range(2*opts.style_interpolate+1):
            style = style1_random - step1*j
            img = decode1(content2[i], style)
            image1.append((img.cpu().data+1)/2)

save_images(opts.output_folder, '1_interpolate_s.jpg', image1, 2*(opts.style_interpolate+1))
del image1

style2_random = torch.randn(*style2[0].size(), device=style2[0].device)
step2 = style2_random / opts.style_interpolate
# within-domain content
image2 = []
for i in range(num_input):
    with torch.no_grad():
        img = decode2(content2[i], style2[i])
        image2.append((img.cpu().data+1)/2)
        for j in range(2*opts.style_interpolate+1):
            style = style2_random - step2*j
            img = decode2(content2[i], style)
            image2.append((img.cpu().data+1)/2)
# cross-domain content
for i in range(num_input):
    with torch.no_grad():
        img = decode2(content1[i], style2[i])
        image2.append((img.cpu().data+1)/2)
        for j in range(2*opts.style_interpolate+1):
            style = style2_random - step2*j
            img = decode1(content1[i], style)
            image2.append((img.cpu().data+1)/2)

save_images(opts.output_folder, '2_interpolate_s.jpg', image2, 2*(opts.style_interpolate+1))
del image2
