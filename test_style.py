from utils import get_config, pytorch03_to_pytorch04
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
parser.add_argument('--input_list', type=str, help="input image list")
parser.add_argument('--style_list', type=str, help="style image list")
parser.add_argument('--output_folder', type=str, help="output image folder")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--num_style',type=int, default=19, help="number of styles to sample")
parser.add_argument('--input_folder', type=str, default=None, help="input image folder")
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
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), 'MUNIT')
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

trainer.to(device)
trainer.eval()
encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode # encode function
encode2 = trainer.gen_b.encode if opts.a2b else trainer.gen_a.encode # encode function
decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode # decode function
decode2 = trainer.gen_a.decode if opts.a2b else trainer.gen_b.decode # decode function
# encode_style = trainer.gen_b.encode if opts.a2b else trainer.gen_a.encode # encode function

transform_list = [transforms.Resize(config['new_size'])]
if opts.centercrop:
    transform_list.append(transforms.CenterCrop((config['new_size'], config['new_size'])))
transform_list.extend([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.Compose(transform_list)

with open(opts.input_list) as f1:
    for j in f1:
        j = j.strip()
        if j.startswith('#'):
            continue
        print("handling: "+j)
        img = Image.open(os.path.join(opts.input_folder, j+'.jpg'))
        img = transform(img)
        img = img.to(device)
        content, _ = encode(img.unsqueeze(0))

        with open(opts.style_list) as f:
            for i in f:
                i = i.strip()
                if i.startswith('#'):
                    continue
                print("  handling: "+i)
                img = Image.open(os.path.join(opts.style_folder, i+'.jpg'))
                img = transform(img)
                img = img.to(device)
                _, style = encode2(img.unsqueeze(0))
                img = decode(content, style)
                img = (img.cpu().data+1)/2
                vutils.save_image(img, os.path.join(opts.output_folder, j+'_'+i+'.jpg'), nrow=1)

# style_random = torch.randn(opts.num_style, style_dim, 1, 1).to(device)

# with open(opts.input_list) as f:
#     for i in f:
#         i = i.strip()
#         if i.startswith('#'):
#             continue
#         print("handling"+i)
#         img = Image.open(os.path.join(opts.input_folder, i+'.jpg'))
#         img = transform(img)
#         img = img.to(device)
#         content, style = encode(img.unsqueeze(0))
#         img = decode2(content, style)
#         img = (img.cpu().data+1)/2
#         vutils.save_image(img, os.path.join(opts.output_folder, i+'_0.jpg'), nrow=1)
#         for j in range(opts.num_style):
#             img_trans = decode(content, style_random[j].unsqueeze(0))
#             img_trans = (img_trans.cpu().data+1)/2
#             vutils.save_image(img_trans, os.path.join(opts.output_folder, i+'_{}.jpg'.format(j+1)), nrow=1)

