from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04
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
parser.add_argument('--config', type=str, help='Path to the config file.')
parser.add_argument('--input_folderA', type=str, help="input image folder of domain A")
parser.add_argument('--input_folderB', type=str, help="input image folder of domain B")
parser.add_argument('--output_folder', type=str, help="output image folder")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--gpu_id', type=int, default=0, help="which gpu to use")

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

loaderA = get_data_loader_folder(opts.input_folderA, 1, False, new_size=config['new_size'], 
                                    crop=True, height=config['new_size'], width=config['new_size'])
loaderB = get_data_loader_folder(opts.input_folderB, 1, False, new_size=config['new_size'], 
                                    crop=True, height=config['new_size'], width=config['new_size'])

content1, style1 = [], []
image1 = []
num_input1 = 0
for data in loaderA:
    num_input1 += 1
    image1.append((data+1)/2)
    data = data.to(device)
    with torch.no_grad():
        content, style = encode1(data)
    content1.append(content)
    style1.append(style)

content2, style2 = [], []
image2 = []
num_input2 = 0
for data in loaderB:
    num_input2 += 1
    image2.append((data+1)/2)
    data = data.to(device)
    with torch.no_grad():
        content, style = encode2(data)
    content2.append(content)
    style2.append(style)

num_input = min(num_input1, num_input2)
if num_input < num_input1:
    content1 = content1[:num_input]
    style1 = style1[:num_input]
if num_input < num_input2:
    content2 = content2[:num_input]
    style2 = style2[:num_input]

# a2b
image1_1, image1_2, image1_2_1 = [], [], []
content1_2, style1_2 = [], []
for i in range(num_input):
    with torch.no_grad():
        # recon
        img = decode1(content1[i], style1[i])
        image1_1.append((img.cpu().data+1)/2)
        # trans
        img = decode2(content1[i], style2[i])
        image1_2.append((img.cpu().data+1)/2)
        # cycle
        content, style = encode2(img)
        img = decode1(content, style1[i])
        content1_2.append(content)
        style1_2.append(style)
        image1_2_1.append((img.cpu().data+1)/2)

print('1.1 Magnitude of content:\nmax:')
content1[0], _ = content1[0].view(-1).sort()
content1[1], _ = content1[1].view(-1).sort()
content1_2[0] = content1_2[0].view(-1)
content1_2[1] = content1_2[1].view(-1)
print(content1[0][-1], content1[1][-1])
print('medium:')
medium = content1[0].size(0) // 2
print(content1[0][medium], content1[1][medium])
print('min:')
print(content1[0][0], content1[1][0])

print('1.2 Difference between cycle content:')
print(torch.mean(torch.abs(content1[0]-content1_2[0])), torch.mean(torch.abs(content1[1]-content1_2[1])))

print('1.3 Difference between different content:')
print(torch.mean(torch.abs(content1[0]-content1[1])), torch.mean(torch.abs(content1_2[0]-content1_2[1])))

print('\n2.1 Magnitude of style:\nmax:')
style1[0], _ = style1[0].view(-1).sort()
style1[1], _ = style1[1].view(-1).sort()
style1_2[0] = style1_2[0].view(-1)
style1_2[1] = style1_2[1].view(-1)
print(style1[0][-1], style1[1][-1])
print('medium:')
print(style1[0][medium], style1[1][medium])
print('min:')
print(style1[0][0], style1[1][0])

print('2.2 Difference between cycle style:')
print(torch.mean(torch.abs(style1[0]-style1_2[0])), torch.mean(torch.abs(style1[1]-style1_2[1])))

print('2.3. Difference between different style:')
print(torch.mean(torch.abs(style1[0]-style1[1])), torch.mean(torch.abs(style1_2[0]-style1_2[1])))

save_images(opts.output_folder, 'a2b.jpg', image1+image1_1+image1_2+image1_2_1, num_input)

# b2a
image2_2, image2_1, image2_1_2 = [], [], []
content2_1, style2_1 = [], []
for i in range(num_input):
    with torch.no_grad():
        # recon
        img = decode2(content2[i], style2[i])
        image2_2.append((img.cpu().data+1)/2)
        # trans
        img = decode1(content2[i], style1[i])
        image2_1.append((img.cpu().data+1)/2)
        # cycle
        content, style = encode1(img)
        img = decode2(content, style2[i])
        content2_1.append(content)
        style2_1.append(style)
        image2_1_2.append((img.cpu().data+1)/2)

print('\n1.1 Magnitude of content:\nmax:')
content2[0], _ = content2[0].view(-1).sort()
content2[1], _ = content2[1].view(-1).sort()
content2_1[0] = content2_1[0].view(-1)
content2_1[1] = content2_1[1].view(-1)
print(content2[0][-1], content2[1][-1])
print('medium:')
print(content2[0][medium], content2[1][medium])
print('min:')
print(content2[0][0], content2[1][0])

print('1.2 Difference between cycle content:')
print(torch.mean(torch.abs(content2[0]-content2_1[0])), torch.mean(torch.abs(content2[1]-content2_1[1])))

print('1.3 Difference between different content:')
print(torch.mean(torch.abs(content2[0]-content2[1])), torch.mean(torch.abs(content2_1[0]-content2_1[1])))

print('\n2.1 Magnitude of content:\nmax:')
style2[0], _ = style2[0].view(-1).sort()
style2[1], _ = style2[1].view(-1).sort()
style2_1[0] = style2_1[0].view(-1)
style2_1[1] = style2_1[1].view(-1)
print(style2[0][-1], style2[1][-1])
print('medium:')
print(style2[0][medium], style2[1][medium])
print('min:')
print(style2[0][0], style2[1][0])

print('2.2 Difference between cycle content:')
print(torch.mean(torch.abs(style2[0]-style2_1[0])), torch.mean(torch.abs(style2[1]-style2_1[1])))

print('2.3 Difference between different content:')
print(torch.mean(torch.abs(style2[0]-style2[1])), torch.mean(torch.abs(style2_1[0]-style2_1[1])))

save_images(opts.output_folder, 'b2a.jpg', image2+image2_2+image2_1+image2_1_2, num_input)
