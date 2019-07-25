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


def get_logits(input_folder, input_list, transform, dis):
    outputs = []
    for i in input_list:
        outputs.append([])
        img = Image.open(os.path.join(opts.input_folder, i+'.jpg'))
        img = transform(img)
        img = img.to(device)
        logits = dis.forward(img.unsqueeze(0))
        for logit in logits:
            outputs[-1].append(torch.mean(logit).data().cpu())
    return outputs

def save_logits(output_folder, logits, remark):
    with open(os.path.join(output_folder, 'logits.txt'), 'a') as f:
        f.write(remark + ':\n')
        for logit in logits:
            f.write(str(logit[0]) + ', ' + str(logit[1]) + ', ' + str(logit[2]) + ', ')
            f.write(str((logit[0]+logit[1]+logit[2])/3) + '\n')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder', help='Path to the config file.')
parser.add_argument('--input_list', type=str, help="input image list")
parser.add_argument('--output_folder', type=str, help="output image folder")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--real_input_folder', type=str, default=None, help="input image folder")
parser.add_argument('--fake_input_folder', type=str, default=None, help="input image folder")
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

trainer = MUNIT_Trainer(config, device)

try:
    state_dict = torch.load(opts.checkpoint)
    trainer.dis_a.load_state_dict(state_dict['a'])
    trainer.dis_b.load_state_dict(state_dict['b'])
except:
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), 'MUNIT')
    trainer.dis_a.load_state_dict(state_dict['a'])
    trainer.dis_b.load_state_dict(state_dict['b'])

dis = trainer.dis_b if opts.a2b else trainer.dis_a

trainer.to(device)
trainer.eval()

transform_list = [transforms.Resize(config['crop_image_width'])]
if opts.centercrop:
    transform_list.append(transforms.CenterCrop((config['crop_image_height'], config['crop_image_width'])))
transform_list.extend([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.Compose(transform_list)

input_list = [ [],[],[] ]
with open(opts.input_list) as f:
    i = -1
    for line in f:
        line = line.strip()
        if line == '':
            continue
        if line.startswith('#'):
            i += 1
            continue
        input_list[i].append(line)

output_real = get_logits(opts.real_input_folder, input_list[0], transform, dis)
output_fake_1 = get_logits(opts.fake_input_folder, input_list[1], transform, dis)
output_fake_2 = get_logits(opts.fake_input_folder, input_list[2], transform, dis)

save_logits(opts.output_folder, output_real, 'Logits of real images')
save_logits(opts.output_folder, output_fake_1, 'Logits of fake images')
save_logits(opts.output_folder, output_fake_2, 'Logits of fake images')