# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


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

import clip
# from torch.utils.tensorboard import SummaryWriter

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

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
    parser.add_argument('--adapter_s', default=1.0, type=float)
    parser.add_argument('--name_fusion_s', default=0.5, type=float)
    parser.add_argument('--nb', action='store_true', help='')
    parser.add_argument('--kd', action='store_true', help='')
    parser.add_argument('--vlm', action='store_true', help='')
    parser.add_argument('--eval_type', default='all', type=str,help="")
    parser.add_argument('--ifsd', default='', help='incremental learning from checkpoint')


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


def main(args):
    # torch.autograd.set_detect_anomaly(True)
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.vlm:
        clip_model, clip_preprocess = clip.load('RN50', device)
        clip_model = torch.nn.parallel.DistributedDataParallel(clip_model, device_ids=[args.gpu])

    else:
        clip_model = None
        clip_preprocess = None

    model, criterion, postprocessors = build_model(args)

    if args.kd:
        model_teacher= build_model_teacher(args)
        model_teacher_without_ddp = model_teacher
    else :
        model_teacher = None

    model.to(device)

    model_without_ddp = model
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(args.vlm,image_set='train', args=args)
    dataset_val = build_dataset(args.vlm,image_set='val', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names)  and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
            
        # check the resumed model
        if args.eval:
            test_stats, coco_evaluator = evaluate(
            model,model_teacher,clip_model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir , eval_type="base"
            )   

    if args.i: 
        
        new = model_adapter(model_without_ddp,args.kd,args.adapter_s)
        model = new.get_model()
        model.to(device)

        if args.kd:
            model_teacher.to(device)

        checkpoint = torch.load(args.ifsd, map_location='cpu')

        print(checkpoint['model']['class_embed.0.weight'].shape)
        print(checkpoint['model']['class_embed.0.bias'].shape)

        if args.eval:
            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        else :
            missing_keys, unexpected_keys = model_without_ddp.load_state_dict( {k:v for k,v in checkpoint['model'].items() if "class_embed" not in k and "class_text_embed" not in k}, strict=False)
            if args.kd:
                # teacher_missing_keys, teacher_unexpected_keys = model_teacher_without_ddp.load_state_dict( {k:v for k,v in checkpoint['model'].items() if "class_embed" not in k}, strict=False)
                teacher_missing_keys, teacher_unexpected_keys = model_teacher_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            for name,params in model_without_ddp.named_parameters():
                if "class_embed_base" in name:
                    # class_embed_base
                    if "weight" in name:
                        if args.two_stage:
                            params[:61,:].data.copy_(checkpoint['model']['class_embed.{}.weight'.format(5)][:61,:])
                            print(name)
                            print(params)
                        else:
                            for i in range (7):
                                if str(i) in name: 
                                    params[:61,:].data.copy_(checkpoint['model']['class_embed.{}.weight'.format(i)][:61,:])
                                    print(name)
                                    print(params)
                    else :
                        if args.two_stage:
                            params[:61].data.copy_(checkpoint['model']['class_embed.{}.bias'.format(5)][:61])
                            print(name)
                            print(params)
                        else:
                            for i in range (7):
                                if str(i) in name: 
                                    params[:61].data.copy_(checkpoint['model']['class_embed.{}.bias'.format(i)][:61])
                                    print(name)
                                    print(params)


        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        
        if args.nb: 
            model_without_ddp.transformer.novel_linear.weight.data = model_without_ddp.transformer.text_linear.weight.data.clone()
            model_without_ddp.transformer.novel_linear.bias.data = model_without_ddp.transformer.text_linear.bias.data.clone()
            if args.two_stage:
                # model_without_ddp.transformer.decoder.class_embed.weight[:61,:].data.copy_(checkpoint['model']['transformer.decoder.class_embed.weight'][:61])
                # model_without_ddp.transformer.decoder.class_embed.bias[:61].data.copy_(checkpoint['model']['transformer.decoder.class_embed.bias'][:61])
                model_without_ddp.transformer.decoder.class_embed.weight[:61,:].data.copy_(checkpoint['model']['class_embed.6.weight'][:61,:])
                model_without_ddp.transformer.decoder.class_embed.bias[:61].data.copy_(checkpoint['model']['class_embed.6.bias'][:61])

        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if args.kd:
            teacher_unexpected_keys = [k for k in teacher_unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
            if len(teacher_missing_keys) > 0:
                print('Teacher Missing Keys: {}'.format(teacher_missing_keys))
            if len(teacher_unexpected_keys) > 0:
                print('Teacher Unexpected Keys: {}'.format(teacher_unexpected_keys))

        param_dicts = [
            {
                "params":
                    [p for n, p in model_without_ddp.named_parameters()
                    if p.requires_grad],
                "lr": args.lr,
            },
        ]
        
        if args.sgd:
            optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                        weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                        weight_decay=args.weight_decay)
    
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
            if args.kd:
                model_teacher = torch.nn.parallel.DistributedDataParallel(model_teacher, device_ids=[args.gpu])
                model_teacher_without_ddp = model_teacher.module

    for name, param in model.named_parameters():
        print("name: ", name)
        print("size: ",param.shape)
        print("requires_grad: ", param.requires_grad)

    if args.kd:
        print("k____d")
        for name, param in model_teacher.named_parameters():
            param.requires_grad = False
            print("name: ", name)
            print("size: ",param.shape)
            print("requires_grad: ", param.requires_grad)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.eval:
        test_stats, coco_evaluator = evaluate(
            model,model_teacher, clip_model,criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir , eval_type="novel",name_fusion_s=args.name_fusion_s
        )
        test_stats, coco_evaluator = evaluate(
            model, model_teacher,clip_model,criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir , eval_type="base",name_fusion_s=args.name_fusion_s
        )
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")

    start_time = time.time()
    iters = 0
    end_iters = 400
    if args.two_stage:end_iters=end_iters*2
    ff = False
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
            train_stats,iter_temp = train_one_epoch(
                model, model_teacher,clip_model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm,name_fusion_s=args.name_fusion_s)
            iters += iter_temp

        if args.i != True:
            lr_scheduler.step()
        else:
            if iters >=end_iters*0.8  and ff == False :
                ff = True
                for _ in range(args.lr_drop):
                    lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            # (epoch + 1 ) % 50 == 0
            if ((epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.epochs) and args.i!=True:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            if iters>=end_iters and args.i:
                    checkpoint_paths.append(output_dir / f'checkpoint_final.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats = {}

        if ((epoch + 1) % 5 == 0 and args.i !=True) or (iters>=end_iters and args.i):
            if args.eval_type == 'all':
                if args.i:
                    test_stats, coco_evaluator = evaluate(
                        model,model_teacher, clip_model,criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir , eval_type="novel",name_fusion_s=args.name_fusion_s
                    )
                test_stats, coco_evaluator = evaluate(
                    model, model_teacher,clip_model,criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir , eval_type="base",name_fusion_s=args.name_fusion_s
                )
                print(iters)

        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #             **{f'test_{k}': v for k, v in test_stats.items()},
        #             'epoch': epoch,
        #             'n_parameters': n_parameters}

        # if args.output_dir and utils.is_main_process():
        #     with (output_dir / "log.txt").open("a") as f:
        #         f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
        if iters >=end_iters and args.i:
            break
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
