from utils import get_config, get_data_loader_folder
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
parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--num_style',type=int, default=3, help="number of styles to sample")
parser.add_argument('--style_folder', type=str, default=None, help="style image folder")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--gpu_id', type=int, default=0, help="which gpu to use")
parser.add_argument('--test_mode', type=str, default='continuity', help='continuity | withindomain')

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

state_dict = torch.load(opts.checkpoint)
trainer.gen_a.load_state_dict(state_dict['a'])
trainer.gen_b.load_state_dict(state_dict['b'])

trainer.to(device)
trainer.eval()

if opts.test_mode == 'continuity':
    encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode
    decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode
    encode_style = trainer.gen_b.encode if opts.a2b else trainer.gen_a.encode

    style_loader = get_data_loader_folder(opts.style_folder, 1, False, new_size=config['new_size'], crop=False)
    style_end = []
    images = []
    for i, data in enumerate(style_loader):
        if i == 2:
            break
        data = data.to(device)
        with torch.no_grad():
            _, style = encode_style(data)
        style_end.append(style)
    assert len(style_end) == 2
    delta = (style_end[1] - style_end[0]) / (opts.num_style-1)

    content_loader = get_data_loader_folder(opts.input_folder, 1, False, new_size=config['new_size'], crop=False)
    num_input = 0
    for data in content_loader:
        images.append((data+1)/2)
        data = data.to(device)
        with torch.no_grad():
            content, _ = encode(data)
        for j in range(opts.num_style):
            style = style_end[0] + j*delta
            with torch.no_grad():
                img = decode(content, style)
            images.append((img.cpu().data+1)/2)
    assert len(images) == (opts.num_style+1)*num_input
    save_images(opts.output_folder, 'interpolation.jpg', images, opts.num_style+1)

elif opts.test_mode == 'withindomain':
    encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode
    decode = trainer.gen_a.decode if opts.a2b else trainer.gen_b.decode

    loader = get_data_loader_folder(opts.input_folder, 1, False, new_size=config['new_size'], crop=False)
    
    images = []
    content_encode, style_encode = [], []
    for data in loader:
        
        data = data.to(device)
        with torch.no_grad():
            content, style = encode_style(data)
        content_encode.append(content)
        style_encode.append(style)

    num_input = len(content_encode)  
    for j in range(num_input):
        content = content_encode[j]
        style = style_encode[j]
        with torch.no_grad():
            img = decode(content, style)
        images.append((img.cpu().data+1)/2)

    assert len(images) == num_input**2
    save_images(opts.output_folder, 'withindomain.jpg', images, num_input)
else:
    sys.exit('ONLY support continuity or withindomain')