# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

import torchvision

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn

import clip


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,vlm=False,nb=False,i=False):
        super().__init__()
        self.vlm =vlm
        self.nb = nb
        self.i = i
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)
        

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if self.vlm:
            self.text_linear = nn.Linear(1024,256)
            if self.nb:
                self.novel_linear = nn.Linear(1024,256)

        if self.vlm:
            clip_model, _ = clip.load('RN50',device='cpu')
            with torch.no_grad():
                text_inputs = []
                text_inputs.append(clip.tokenize("background"))
                keys_list = list(dataset_classes_name_list.keys())
                num = 60 if self.nb or self.i!=True else 80

                for v in keys_list[:num]:
                    s = f"A photo of a/an {v} in the scene"
                    print(s)
                    text_inputs.append(clip.tokenize(s))
                text_inputs = torch.cat(text_inputs) 
            self.class_text_embed =nn.Parameter(clip_model.encode_text(text_inputs),requires_grad=False)
            
            if self.nb:
                text_inputs_novel = []
                keys_list = list(dataset_classes_name_list_novel.keys())
                for v in keys_list[:20]:
                    s = f"A photo of a/an {v} in the scene"
                    print(s)
                    text_inputs_novel.append(clip.tokenize(s))
                text_inputs_novel = torch.cat(text_inputs_novel)

                self.class_text_embed_novel =nn.Parameter(clip_model.encode_text(text_inputs_novel),requires_grad=False)

            self.feature_attention = nn.MultiheadAttention(1024,8,batch_first=True)
            self.feature_norm = nn.LayerNorm(1024)

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None,text_embedding=None,padding_list=None):
        assert self.two_stage or query_embed is not None
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        
        if self.vlm:
            batch_size = srcs[0].shape[0]
            vlm_mask = torch.zeros(text_embedding.shape[0],text_embedding.shape[1], dtype=torch.bool ,device=text_embedding.device)
            for i, count in enumerate(padding_list):
                vlm_mask[i, :count] = True
            vlm_mask = vlm_mask.unsqueeze(-1).expand(-1, -1, text_embedding.shape[2])
            


            text_embedding = text_embedding.to(torch.float32)
            class_text_embed = self.class_text_embed.unsqueeze(0).expand(batch_size,-1,-1)
            text_embedding_strengthening,_ = self.feature_attention(text_embedding,class_text_embed,class_text_embed)
            text_embedding_base = text_embedding + text_embedding_strengthening
            text_embedding_base = self.feature_norm(text_embedding_base)

            if self.nb:
                class_text_embed_novel = self.class_text_embed_novel.unsqueeze(0).expand(batch_size,-1,-1)
                text_embedding_strengthening_novel,_ = self.feature_attention(text_embedding,class_text_embed_novel,class_text_embed_novel)
                text_embedding_novel = text_embedding + text_embedding_strengthening_novel
                text_embedding_novel = self.feature_norm(text_embedding_novel)
            
            #text_embedding_base = torch.where(vlm_mask, text_embedding_base, torch.tensor(0.0, device=text_embedding.device))
            #text_embedding_base = torch.sum(text_embedding_base, dim=1)
            text_embedding_base = self.text_linear(text_embedding_base)
            #padding_list = padding_list.clone().detach().view(-1, 1).expand(-1, 256)
            #padding_list = padding_list.clone().detach().view(-1, 1, 1).expand(-1, text_embedding_base.shape[1], 256)
            #text_embedding_base = text_embedding_base / padding_list

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # prepare input for decoder
        # print(memory.shape)
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            if self.i:
                enc_outputs_class = self.decoder.class_embed(output_memory)
            else:
                enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
                # enc_outputs_class = self.decoder.class_embed(output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:

            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)

            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        if self.vlm: 
            #text_embedding_base = text_embedding_base.unsqueeze(1).expand(-1,tgt.shape[1],-1)
            if self.nb:
                #text_embedding_novel = torch.where(vlm_mask, text_embedding_novel, torch.tensor(0.0 , device=text_embedding.device))
                #text_embedding_novel = torch.sum(text_embedding_novel, dim=1)
                text_embedding_novel = self.novel_linear(text_embedding_novel)
                #padding_list = padding_list.clone().detach().view(-1, 1, 1).expand(-1, text_embedding_novel.shape[1], 256)
                #text_embedding_novel = text_embedding_novel / padding_list
                #text_embedding_novel = text_embedding_novel.unsqueeze(1).expand(-1,tgt.shape[1],-1)
            
            tgt = torch.cat([tgt, text_embedding_base], dim=1)
            #tgt = tgt + text_embedding_base
            if self.nb:
                tgt_novel = torch.cat([tgt, text_embedding_novel], dim=1)
                #tgt_novel = tgt_novel + text_embedding_novel

        # decoder
        if self.i:
            hs, inter_references = self.decoder(tgt, reference_points, memory,
                                                    spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten,novel=False)
            if self.nb:
                hs_novel,_ = self.decoder(tgt_novel, reference_points, memory,
                                                    spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten,novel=True)
        else:
            hs, inter_references = self.decoder(tgt, reference_points, memory,
                                                    spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten,novel="base")
        inter_references_out = inter_references

        if self.two_stage:
            if self.nb:
                return hs,hs_novel, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact

        # return hs, init_reference_out, inter_references_out, None, None
        if self.nb:
            return hs, hs_novel, init_reference_out, inter_references_out, None, None
        return hs, init_reference_out, inter_references_out, None, None
        

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()


        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        


    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
    def forward_ffn_text(self, tgt):
        tgt2 = self.linear_2_text(self.dropout3_text(self.activation_text(self.linear1_text(tgt))))
        tgt = tgt + self.dropout4_text(tgt2)
        tgt = self.norm3_text(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        #print("forward:", tgt.shape)
        num_to_add = tgt.shape[1] - query_pos.shape[1]
        additional_query_pos = torch.randn(query_pos.shape[0], num_to_add, query_pos.shape[2], device=query_pos.device)  # [2, 1, 1024]
        # 拼接到 query_pos
        query_pos = torch.cat([query_pos, additional_query_pos], dim=1)

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt = tgt[:, :300, :]
        query_pos = query_pos[:, :300, :]

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        
        tgt= self.forward_ffn(tgt)
    
        # tgt = self.forward_ffn(tgt)
        
        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)

        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None,novel=False):
        output = tgt
        intermediate = []
        intermediate_reference_points = []
         
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            if novel == "base":
                output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes,src_level_start_index, src_padding_mask)
            else:
                output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes,src_level_start_index ,novel, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate),torch.stack(intermediate_reference_points)

        return output, reference_points

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


# class Adapter(nn.Module):
#     """
#     The adapters first project the original
#     d-dimensional features into a smaller dimension, m, apply
#     a nonlinearity, then project back to d dimensions.
#     """
#     def __init__(self, size = 128, model_dim = 256):
#         super().__init__()
#         self.adapter_block = nn.Sequential(
#             nn.Linear(model_dim, size),
#             nn.ReLU(),
#             nn.Linear(size, model_dim)
#         )

#     def forward(self, x):

#         ff_out = self.adapter_block(x)

#         # Skip connection
#         adapter_out = (ff_out * 0.1) + x

#         return adapter_out

# class Branch_Adapter(nn.Module):
#     """
#     The adapters first project the original
#     d-dimensional features into a smaller dimension, m, apply
#     a nonlinearity, then project back to d dimensions.
#     """
#     def __init__(self, size = 128, model_dim = 256):
#         super().__init__()
#         # self.adapter_block1= nn.Sequential(
#         #     nn.Linear(model_dim, size),
#         #     nn.ReLU(),
#         #     nn.Linear(size, model_dim)
#         # )
#         self.adapter_block = nn.Sequential(
#             nn.Linear(model_dim, size),
#             nn.ReLU(),
#             nn.Linear(size, model_dim)
#         )
#         self.norm = nn.LayerNorm(model_dim)

#     def forward(self, x1, x2):

#         # ff_out1 = self.adapter_block1(x1)
#         ff_out = self.adapter_block(x1)

#         # Skip connection
#         adapter_out = self.norm( ff_out*0.1 + x2)

#         return adapter_out


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        vlm = args.vlm,
        nb = args.nb,
        i = args.i
        )


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
dataset_classes_name_list = {v: k for k, v in dataset_classes_name_list.items()}

dataset_classes_name_list_novel = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorbike",
    5: "aeroplane",
    6: "bus",
    7: "train",
    8: "boat",
    9: "bird",
    10: "cat",
    11: "dog",
    12: "horse",
    13: "sheep",
    14: "cow",
    15: "bottle",
    16: "chair",
    17: "sofa",
    18: "pottedplant",
    19: "diningtable",
    20: "tvmonitor",
}
dataset_classes_name_list_novel = {v: k for k, v in dataset_classes_name_list_novel.items()}

# prompts = [
#     'There is {article} {category} in the scene.',
#     'There is the {category} in the scene.',
#     'a photo of {article} {category} in the scene.',
#     'a photo of the {category} in the scene.',
#     'a photo of one {category} in the scene.',
#     'itap of {article} {category}.',
#     'itap of my {category}.',
#     'itap of the {category}.',
#     'a photo of {article} {category}.',
#     'a photo of my {category}.',
#     'a photo of the {category}.',
#     'a photo of one {category}.',
#     'a photo of many {category}.',
#     'a good photo of {article} {category}.',
#     'a good photo of the {category}.',
#     'a bad photo of {article} {category}.',
#     'a bad photo of the {category}.',
#     'a photo of a nice {category}.',
#     'a photo of the nice {category}.',
#     'a photo of a cool {category}.',
#     'a photo of the cool {category}.',
#     'a photo of a weird {category}.',
#     'a photo of the weird {category}.',
#     'a photo of a small {category}.',
#     'a photo of the small {category}.',
#     'a photo of a large {category}.',
#     'a photo of the large {category}.',
#     'a photo of a clean {category}.',
#     'a photo of the clean {category}.',
#     'a photo of a dirty {category}.',
#     'a photo of the dirty {category}.',
#     'a bright photo of {article} {category}.',
#     'a bright photo of the {category}.',
#     'a dark photo of {article} {category}.',
#     'a dark photo of the {category}.',
#     'a photo of a hard to see {category}.',
#     'a photo of the hard to see {category}.',
#     'a low resolution photo of {article} {category}.',
#     'a low resolution photo of the {category}.',
#     'a cropped photo of {article} {category}.',
#     'a cropped photo of the {category}.',
#     'a close-up photo of {article} {category}.',
#     'a close-up photo of the {category}.',
#     'a jpeg corrupted photo of {article} {category}.',
#     'a jpeg corrupted photo of the {category}.',
#     'a blurry photo of {article} {category}.',
#     'a blurry photo of the {category}.',
#     'a pixelated photo of {article} {category}.',
#     'a pixelated photo of the {category}.',
#     'a black and white photo of the {category}.',
#     'a black and white photo of {article} {category}.',
#     'a plastic {category}.',
#     'the plastic {category}.',
#     'a toy {category}.',
#     'the toy {category}.',
#     'a plushie {category}.',
#     'the plushie {category}.',
#     'a cartoon {category}.',
#     'the cartoon {category}.',
#     'an embroidered {category}.',
#     'the embroidered {category}.',
#     'a painting of the {category}.',
#     'a painting of a {category}.',
# ]

# def choose_article(category):
#     if category[0].lower() in 'aeiou':
#         return 'an'
#     else:
#         return 'a'