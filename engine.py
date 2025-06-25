# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher

from util.misc import NestedTensor, nested_tensor_from_tensor_list

import clip

def train_one_epoch(model: torch.nn.Module, model_teacher: torch.nn.Module,vlm_model: torch.nn.Module,criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,name_fusion_s=0.5):
    model.train()
    criterion.train()

    if vlm_model is not None :vlm_model.eval()
    if model_teacher is not None : model_teacher.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    if vlm_model is not None :
        with torch.no_grad():
            bg_feature =  vlm_model.module.encode_text(clip.tokenize("background").to(device))
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        text_features = []
        text_name_features = []
        features = []
        if vlm_model is not None :
            with torch.no_grad():
                padding_list = torch.stack([torch.tensor(len(t['text'])) for t in targets]).to(device)
                padding_len = max(padding_list)
                for i in range(len(targets)):

                    text = targets[i]['text']
                    text_name = targets[i]['text_name']

                    text_feature = vlm_model.module.encode_text(text)
                    text_name_feature = vlm_model.module.encode_text(text_name)
                    
                    if len(text_name) < padding_len:
                        bgs = bg_feature.expand(padding_len-len(text_name),-1)
                        text_feature = torch.cat((text_feature,bgs),dim=0)
                        text_name_feature = torch.cat((text_name_feature,bgs),dim=0)

                    text_features.append(text_feature)
                    text_name_features.append(text_name_feature)
                text_features = torch.stack(text_features)
                
                text_name_features = torch.stack(text_name_features)
                
                features = torch.add(text_features*(1-name_fusion_s),text_name_features*name_fusion_s)
                
        outputs = model(features,padding_list,samples)

        if model_teacher is not None : 
            teacher_output = model_teacher(features,padding_list,samples)
            targets = criterion.delete_boxes(teacher_output,targets,base_label=True)
            for i in range(len(targets)):
                targets[i].update( {'feature_map': [ map[i] for map in teacher_output['feature_map_outputs']]})

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        samples, targets  = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()} ,len(data_loader)


@torch.no_grad()
def evaluate(model, model_teacher,vlm_model,criterion, postprocessors, data_loader, base_ds, device, output_dir,eval_type,name_fusion_s):
    model.eval()
    if model_teacher is not None:
        model_teacher.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types,eval_type)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    if vlm_model is not None :
        with torch.no_grad():
            bg_feature =  vlm_model.module.encode_text(clip.tokenize("background").to(device))

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        text_features = []
        text_name_features = []
        features = []
        # image_features = []
        if vlm_model is not None :
            with torch.no_grad():
                # srcs, masks = samples.decompose()
                padding_list = torch.stack([torch.tensor(len(t['text'])) for t in targets]).to(device)
                padding_len = max(padding_list)
                for i in range(len(targets)):

                    text = targets[i]['text']
                    text_name = targets[i]['text_name']

                    text_feature = vlm_model.module.encode_text(text)
                    text_name_feature = vlm_model.module.encode_text(text_name)

                    if len(text_name) < padding_len:
                        bgs = bg_feature.expand(padding_len-len(text_name),-1)
                        text_feature = torch.cat((text_feature,bgs),dim=0)
                        text_name_feature = torch.cat((text_name_feature,bgs),dim=0)

                    text_features.append(text_feature)
                    text_name_features.append(text_name_feature)

                text_features = torch.stack(text_features)
                text_name_features = torch.stack(text_name_features)
                features = torch.add(text_features*(1-name_fusion_s),text_name_features*name_fusion_s)
        
        outputs = model(features,padding_list,samples)
        
        if model_teacher is not None : 
            teacher_output = model_teacher(features,padding_list,samples)
            targets = criterion.delete_boxes(teacher_output,targets,base_label=False)
            for i in range(len(targets)):
                targets[i].update( {'feature_map': [ map[i] for map in teacher_output['feature_map_outputs']]})

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
