import torch
from PIL import Image, ImageDraw,ImageFont
import torchvision.transforms as T

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model,build_model_teacher

from models.adapter import model_adapter

from torchvision import transforms

from PIL import Image

import clip
import os 


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
    parser.add_argument('--onebyone', action='store_true', help='')
    parser.add_argument('--nb', action='store_true', help='')
    parser.add_argument('--kd', action='store_true', help='')
    parser.add_argument('--two_stage_base', action='store_true', help='')
    parser.add_argument('--cor', action='store_true', help='')
    parser.add_argument('--vlm', action='store_true', help='')
    parser.add_argument('--vlm_withroi', action='store_true', help='')
    parser.add_argument('--vlm_memory', action='store_true', help='')
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
    parser.add_argument('--cor_loss_coef', default=1, type=float)
    parser.add_argument('--vlm_loss_coef', default=5, type=float)
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
dataset_classes_name_list = {
    1: "truck",
    2: "traffic light",
    3: "fire hydrant",
    4: "stop sign",
    5: "parking meter",
    6: "bench",
    7: "elephant",
    8: "bear",
    9: "zebra",
    10: "giraffe",
    11: "backpack",
    12: "umbrella",
    13: "handbag",
    14: "tie",
    15: "suitcase",
    16: "frisbee",
    17: "skis",
    18: "snowboard",
    19: "sports ball",
    20: "kite",
    21: "baseball bat",
    22: "baseball glove",
    23: "skateboard",
    24: "surfboard",
    25: "tennis racket",
    26: "wine glass",
    27: "cup",
    28: "fork",
    29: "knife",
    30: "spoon",
    31: "bowl",
    32: "banana",
    33: "apple",
    34: "sandwich",
    35: "orange",
    36: "broccoli",
    37: "carrot",
    38: "hot dog",
    39: "pizza",
    40: "donut",
    41: "cake",
    42: "bed",
    43: "toilet",
    44: "laptop",
    45: "mouse",
    46: "remote",
    47: "keyboard",
    48: "cell phone",
    49: "microwave",
    50: "oven",
    51: "toaster",
    52: "sink",
    53: "refrigerator",
    54: "book",
    55: "clock",
    56: "vase",
    57: "scissors",
    58: "teddy bear",
    59: "hair drier",
    60: "toothbrush",
    61: "person",
    62: "bicycle",
    63: "car",
    64: "motorbike",
    65: "aeroplane",
    66: "bus",
    67: "train",
    68: "boat",
    69: "bird",
    70: "cat",
    71: "dog",
    72: "horse",
    73: "sheep",
    74: "cow",
    75: "bottle",
    76: "chair",
    77: "sofa",
    78: "pottedplant",
    79: "diningtable",
    80: "tvmonitor",
}
    
def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def preprocess(image):
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def detr_inference(image, model, threshold=0.5,device=None,features=None,padding_list=None):
    # 轉換並增加批次維度
    img_tensor = image.unsqueeze(0)

    # 將圖片移到 GPU 上
    img_tensor = img_tensor.to(device)

    # 獲取預測
    with torch.no_grad():
        outputs = model(features,padding_list,img_tensor)

    # 將輸出移到 CPU 上進行後續處理
    return outputs

# def draw_boxes(image, boxes, labels, scores, colors='red', width=2):
#     draw = ImageDraw.Draw(image)
#     for box, label, score in zip(boxes, labels, scores):
#         if score > 0.3:
#             box = box * torch.tensor([image.width, image.height, image.width, image.height], dtype=torch.float32)
#             label = label.item()
#             print(label)
#             class_name = dataset_classes_name_list.get(label, 'Unknown')
#             if label > 60 :
#                 draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline='blue', width=width)
#                 draw.text((box[0], box[1]), f'{class_name}', fill='blue')
#             else:
#                 draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=colors, width=width)
#                 draw.text((box[0], box[1]), f'{class_name}', fill=colors)
#     return image

def draw_boxes(image, boxes, labels, scores, colors='red', width=2):
    draw = ImageDraw.Draw(image,'RGBA')
    # font = ImageFont.load_default()   # 使用預設字體，您也可以指定自己的字體
    # font = font.font_variant(size=16)
    font = ImageFont.truetype('/data3/age73423/ifsod/Deformable-DETR/Arial.ttf', 24) 

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.3:
            box = box * torch.tensor([image.width, image.height, image.width, image.height], dtype=torch.float32)
            label = label.item()
            class_name = dataset_classes_name_list.get(label, 'Unknown')

            # 計算文字大小
            text_size = draw.textsize(class_name, font=font)

            # 繪製黑色背景
            if label > 60:
                text_color, box_color = colors, colors
            else:
                text_color, box_color = 'blue', 'blue'
                

            # 計算背景位置，並留出一點邊緣
            background_top_left = (box[0], box[1] - text_size[1] - 4)
            background_bottom_right = (box[0] + text_size[0] + 4, box[1])

            # 繪製文字背景
            draw.rectangle([background_top_left, background_bottom_right], fill=(0, 0, 0, 128))

            # 繪製邊界框
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=box_color, width=width)

            # 繪製文字
            draw.text((box[0] + 2, box[1] - text_size[1] - 4), class_name, fill=text_color, font=font)

    return image

def box_cxcywh_to_xyxy(x,device):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1).to(device)
 
def rescale_bboxes(out_bbox, size,device):
    b = box_cxcywh_to_xyxy(out_bbox,device)
    # 保证都使用显卡处理数据
    return b


def main(args):
    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.vlm:
        clip_model, clip_preprocess = clip.load('RN50', device)
    
    # 加載模型
    model = model,_,_ = build_model(args,clip_model=clip_model,clip_preprocess=clip_preprocess)
    new = model_adapter(model,args.kd)
    model = new.get_model()
    model.to(device)
    checkpoint = torch.load('/data1/age73423/ckpt_dir/6/checkpoint0049.pth', map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    print(missing_keys)
    print(unexpected_keys)

    tran = transforms.Compose([
    transforms.Resize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # image,target = dataset_val.__getitem__(i)
    image = load_image("/data3/age73423/ifsod/Deformable-DETR/IMG_6085.JPG")
    transformed_image = tran(image)
    filepath = "/data3/age73423/ifsod/Deformable-DETR/example_.json"
    f = open(filepath)
    jf = json.load(f)
    st = jf['candidates'][0]['content']['parts'][0]['text']
    line = st.split("\n")
    line = [l for l in line if len(l) > 0]
    line = [l[1:-1] for l in line if l[-1] == '|']
                 
    line_name = [l.split("|", 1)[0].replace("|", "") for l in line]
    text_name = clip.tokenize(line_name[2:])
                
    line_text = [l.replace("|", "") for l in line]
    text = clip.tokenize(line_text[2:],context_length=77, truncate=True)
            
    text_feature = clip_model.encode_text(text.to(device))
    text_name_feature = clip_model.encode_text(text_name.to(device))
    features = (torch.add(text_feature,text_name_feature) / 2.0).unsqueeze(0)
    print(features.shape)
    padding_list = torch.stack([torch.tensor(len(text))]).to(device)
    print(padding_list.shape)

    # 進行推理
    outputs = detr_inference(transformed_image, model,device=device,features=features,padding_list=padding_list)

    # 繪製預測框
    scores = outputs['pred_logits'][0,:,:].cpu().sigmoid().max(-1)[0]
    labels = outputs['pred_logits'][0,:,:].cpu().sigmoid().max(-1)[1]
    boxes = outputs['pred_boxes'][0,:,:].cpu()
        
    print(scores.shape)
    print(labels.shape)
    print(boxes.shape)

    boxes = rescale_bboxes(boxes,image.size,device).cpu()

    # 視覺化預測
    result_image = draw_boxes(image.copy(), boxes, labels, scores)


    # 顯示圖片
    result_image.save('/data3/age73423/ifsod/Deformable-DETR/example_image_{}.jpg'.format(0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
