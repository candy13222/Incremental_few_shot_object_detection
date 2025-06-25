import time

import os 
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
import torchvision.transforms as T
import torchvision
from util import box_ops
import base64
import requests
torch.set_grad_enabled(False)

import json
 
from models import build_model
from torch.utils.data import DataLoader
from datasets import build_dataset, get_coco_api_from_dataset

from models.adapter import model_adapter
import argparse
 
from torch.nn.functional import dropout,linear,softmax
 
# ============================================== #

from torchcam.utils import overlay_mask
from torchvision.transforms.functional import normalize, resize, to_pil_image
 
# def get_args_parser():
#     parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
#     parser.add_argument('--lr', default=1e-4, type=float)
#     parser.add_argument('--lr_backbone', default=1e-5, type=float)
#     parser.add_argument('--batch_size', default=1, type=int)
#     parser.add_argument('--weight_decay', default=1e-4, type=float)
#     parser.add_argument('--epochs', default=300, type=int)
#     parser.add_argument('--lr_drop', default=200, type=int)
#     parser.add_argument('--clip_max_norm', default=0.1, type=float,
#                         help='gradient clipping max norm')
 
#     # Model parameters
#     parser.add_argument('--frozen_weights', type=str, default=None,
#                         help="Path to the pretrained model. If set, only the mask head will be trained")
#     # * Backbone
#     parser.add_argument('--backbone', default='resnet50', type=str,
#                         help="Name of the convolutional backbone to use")
#     parser.add_argument('--dilation', action='store_true',
#                         help="If true, we replace stride with dilation in the last convolutional block (DC5)")
#     parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
#                         help="Type of positional embedding to use on top of the image features")
 
#     # * Transformer
#     parser.add_argument('--enc_layers', default=6, type=int,
#                         help="Number of encoding layers in the transformer")
#     parser.add_argument('--dec_layers', default=6, type=int,
#                         help="Number of decoding layers in the transformer")
#     parser.add_argument('--dim_feedforward', default=2048, type=int,
#                         help="Intermediate size of the feedforward layers in the transformer blocks")
#     parser.add_argument('--hidden_dim', default=256, type=int,
#                         help="Size of the embeddings (dimension of the transformer)")
#     parser.add_argument('--dropout', default=0.1, type=float,
#                         help="Dropout applied in the transformer")
#     parser.add_argument('--nheads', default=8, type=int,
#                         help="Number of attention heads inside the transformer's attentions")
#     parser.add_argument('--num_queries', default=100, type=int,
#                         help="Number of query slots")  # 论文中对象查询为100
#     parser.add_argument('--pre_norm', action='store_true')
 
#     # * Segmentation
#     parser.add_argument('--masks', action='store_true',
#                         help="Train segmentation head if the flag is provided")
 
#     # Loss
#     parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
#                         help="Disables auxiliary decoding losses (loss at each layer)")
#     # * Matcher
#     parser.add_argument('--set_cost_class', default=1, type=float,
#                         help="Class coefficient in the matching cost")
#     parser.add_argument('--set_cost_bbox', default=5, type=float,
#                         help="L1 box coefficient in the matching cost")
#     parser.add_argument('--set_cost_giou', default=2, type=float,
#                         help="giou box coefficient in the matching cost")
#     # * Loss coefficients
#     parser.add_argument('--mask_loss_coef', default=1, type=float)
#     parser.add_argument('--dice_loss_coef', default=1, type=float)
#     parser.add_argument('--bbox_loss_coef', default=5, type=float)
#     parser.add_argument('--giou_loss_coef', default=2, type=float)
#     parser.add_argument('--eos_coef', default=0.1, type=float,
#                         help="Relative classification weight of the no-object class")
 
#     # dataset parameters
#     parser.add_argument('--dataset_file', default='coco')
#     parser.add_argument('--coco_path', default='', type=str)
#     parser.add_argument('--coco_panoptic_path', type=str)
#     parser.add_argument('--remove_difficult', action='store_true')
 
#     parser.add_argument('--output_dir', default='E:\project_yd\paper_sci_one_yd\Transformer\DETR\detr\\runs\\train',
#                         help='path where to save, empty for no saving')
#     parser.add_argument('--device', default='cuda',
#                         help='device to use for training / testing')
#     parser.add_argument('--seed', default=42, type=int)
 
#     # ============================================================================= #
#     parser.add_argument('--resume', default='', help='resume from checkpoint')
#     # ============================================================================= #
 
#     parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
#                         help='start epoch')
#     parser.add_argument('--eval', action='store_true')
#     parser.add_argument('--num_workers', default=2, type=int)
 
#     # distributed training parameters
#     parser.add_argument('--world_size', default=1, type=int,
#                         help='number of distributed processes')
#     parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
#     return parser

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--i', action='store_true', help='')
    parser.add_argument('--kd', action='store_true', help='')
    parser.add_argument('--onebyone', action='store_true', help='')
    parser.add_argument('--two_stage_base', action='store_true', help='')
    parser.add_argument('--nb', action='store_true', help='')
    parser.add_argument('--eval_type', default='all', type=str,help="")
    parser.add_argument('--ifsod', default='', help='incremental learning from checkpoint')


    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--kd_loss_coef', default=1, type=float)
    parser.add_argument('--kd_class_loss_coef', default=4, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--dataset_seed', default=0,type=int)
    parser.add_argument('--shot', default=10,type=int)
    parser.add_argument('--per_class', default=20,type=int)
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./out',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser
 
class Main():
    def m(self):
        parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
        args = parser.parse_args()
        dataset_train = build_dataset(vlm_model=False,image_set='train', args=args)
        print(len(dataset_train))

        to_pil = T.ToPILImage()
        KEY = "AIzaSyBf6rBYWuIsNgdoBGsJ7B3fIsKZZgiw7ps"
        # =============================================== #
        i = 0
        count = 0 
        pass_id = 0 
        pass_list = []
        os.mkdir("train_json_new_5shot/train_json_{}shot_seed{}".format(str(args.shot),str(args.dataset_seed)))

        while i < len(dataset_train):
            im,tgt =dataset_train.__getitem__(i)

            filepath = "train_json_new_5shot/train_json_{}shot_seed{}/{}.json".format(str(args.shot),str(args.dataset_seed),str(int(tgt['image_id'].cpu())).zfill(12))

            # 檢查檔案是否存在
            if os.path.isfile(filepath) or i in pass_list:
                i += 1
                continue
            else:
                if pass_id == i:
                    count +=1
                    if count >=5:
                        pass_list.append(pass_id)
                        pass_id = 0
                        count =0
                        continue
                else:
                    pass_id = i
                    print(i)
                    print(tgt['image_id'].cpu())
                    print(filepath)

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            for j in range(3):
                im[j] = im[j] * std[j] + mean[j]

            ig = to_pil(im)

            ig.save('./temp_image/{}.jpg'.format(i))

            def encode_image(image_path):
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')

            # Path to your image
            image_path = './temp_image/{}.jpg'.format(i)

            # Getting the base64 string
            base64_image = encode_image(image_path)
     

            url = f'https://generativelanguage.googleapis.com/v1/models/gemini-pro-vision:generateContent?key={KEY}'
            headers = {'Content-Type': 'application/json'}
            data = {
                "contents": [
                    {
                        "parts": [
                            # {"text": "Below are the category names and corresponding numbers of the target objects: {1: 'truck', 2: 'traffic light', 3: 'fire hydrant', 4: 'stop sign', 5: 'parking meter', 6: 'bench', 7: 'elephant', 8: 'bear', 9: 'zebra', 10: 'giraffe', 11: 'backpack', 12: 'umbrella', 13: 'handbag', 14: 'tie', 15: 'suitcase', 16: 'frisbee', 17: 'skis', 18: 'snowboard', 19: 'sports ball', 20: 'kite', 21: 'baseball bat', 22: 'baseball glove', 23: 'skateboard', 24: 'surfboard', 25: 'tennis racket', 26: 'wine glass', 27: 'cup', 28: 'fork', 29: 'knife', 30: 'spoon', 31: 'bowl', 32: 'banana', 33: 'apple', 34: 'sandwich', 35: 'orange', 36: 'broccoli', 37: 'carrot', 38: 'hot dog', 39: 'pizza', 40: 'donut', 41: 'cake', 42: 'bed', 43: 'toilet', 44: 'tv', 45: 'laptop', 46: 'mouse', 47: 'keyboard', 48: 'cell phone', 49: 'microwave', 50: 'oven', 51: 'toaster', 52: 'sink', 53: 'refrigerator', 54: 'book', 55: 'clock', 56: 'vase', 57: 'scissors', 58: 'teddy bear', 59: 'hair drier', 60: 'toothbrush'}tell me the target category number that appears in the picture. Only reply with the numbers, do not reply with any other content."},
                            # {"text": "List the objects and their corresponding locations  in the image."},
                            # {"text": "What are the objects in the picture? What are their corresponding positions? Use a table to present object names and their corresponding positions"},
                            {"text": "Detect every object in the image and describe its characteristics, such as shape, contour, color, texture, components, and feature points, as well as the relative positions in the image. Please present the results in a table format in English, and the description of each object should be within 77 tokens." },
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": base64_image
                                },
                            }
                        ]
                    },
                    
                ],
                "generationConfig": {
                                    "temperature": "0.9"
                                }
                
            }

            response = requests.post(url, headers=headers, json=data)


            json_object = json.dumps(response.json())

            # print(response.json())
            if 'candidates' in response.json().keys():
                print(response.json()['candidates'][0]['content']['parts'][0]['text'])
            else:
                print('error')
                continue

            with open("train_json_new_5shot/train_json_{}shot_seed{}/{}.json".format(str(args.shot),str(args.dataset_seed),str(int(tgt['image_id'].cpu())).zfill(12)), "w") as outfile:
                outfile.write(json_object)
            
            i+=1
        
        print(pass_list)

        path = "train_json_new_5shot/train_json_{}shot_seed{}/pass_list.txt".format(str(args.shot),str(args.dataset_seed))
        f = open(path,'w')
        f.write(str(pass_list))
        f.close()

main = Main()
main.m()