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
parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--num_style',type=int, default=3, help="number of styles to sample")
parser.add_argument('--style_folder', type=str, default=None, help="style image folder")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--gpu_id', type=int, default=0, help="which gpu to use")
parser.add_argument('--blank_img', type=str, help='the blank image for display (if needed) when style transfer')
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
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), opts.trainer)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

trainer.to(device)
trainer.eval()
encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode # encode function
decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode # decode function
encode_style = trainer.gen_b.encode if opts.a2b else trainer.gen_a.encode # encode function

get_loader = get_data_loader_folder_centercrop if opts.centercrop else get_data_loader_folder

# random style
content_loader = get_loader(opts.input_folder, 1, False, new_size=config['new_size'], 
                            crop=True, height=config['new_size'], width=config['new_size'])
content_encode = []

style_random = torch.randn(opts.num_style, style_dim, 1, 1).to(device)
images_random = []

for data in content_loader:
    images_random.append((data.data+1)/2)
    data = data.to(device)
    with torch.no_grad():
        content, _ = encode(data)
    content_encode.append(content)

num_input = len(content_encode)

for i in range(opts.num_style):
    style = style_random[i].unsqueeze(0)
    for j in range(num_input):
        content = content_encode[j]
        with torch.no_grad():
            img = decode(content, style)
        images_random.append((img.cpu().data+1)/2)

assert len(images_random) == num_input*(1+opts.num_style)
save_images(opts.output_folder, 'random.jpg', images_random, num_input)

# encoded style
if opts.style_folder is None:
    sys.exit(0)

images_encode = []

style_loader = get_loader(opts.style_folder, 1, False, new_size=config['new_size'], 
                            crop=True, height=config['new_size'], width=config['new_size'])

blank_img = Image.open(opts.blank_img).resize((config['new_size'], config['new_size']))
to_tensor = transforms.ToTensor()
blank_img = to_tensor(blank_img).unsqueeze(0)
images_encode.append(blank_img)
for data in content_loader:
    images_encode.append((data.data+1)/2)

num_style = 0
for data in style_loader:
    num_style += 1
    images_encode.append((data.data+1)/2)
    
    data = data.to(device)
    with torch.no_grad():
        _, style = encode_style(data)

    for j in range(num_input):
        content = content_encode[j]
        with torch.no_grad():
            img = decode(content, style)
        images_encode.append((img.cpu().data+1)/2)

assert len(images_encode) == (1+num_input)*(1+num_style)
save_images(opts.output_folder, 'encode.jpg', images_encode, num_input+1)

