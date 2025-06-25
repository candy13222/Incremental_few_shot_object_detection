import torch
import torch.nn.functional as F
from torch import nn
import math

class Adapter(nn.Module):
    """
    The adapters first project the original
    d-dimensional features into a smaller dimension, m, apply
    a nonlinearity, then project back to d dimensions.
    """
    def __init__(self, size = 128, model_dim = 256, s=1.0):
        super().__init__()
        self.adapter_block = nn.Sequential(
            nn.Linear(model_dim, size),
            nn.ReLU(),
            nn.Linear(size, model_dim)
        )
        self.s = s
        print("adapter_s : " ,self.s)


    def forward(self, x):

        ff_out = self.adapter_block(x)
        adapter_out = (ff_out * self.s) + x

        return adapter_out
    
class encode_Adaptered(nn.Module):
    def __init__(self,orig_block):
        super().__init__()
        self.adapter2 = Adapter()
        # self attention
        self.self_attn = orig_block.self_attn
        self.dropout1 =  orig_block.dropout1
        self.norm1 = orig_block.norm1

        # ffn
        self.linear1 = orig_block.linear1
        self.activation = orig_block.activation
        self.dropout2 =  orig_block.dropout2
        self.linear2 = orig_block.linear2
        self.dropout3 =  orig_block.dropout3
        self.norm2 = orig_block.norm2

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src3 = self.adapter2(self.dropout3(src2))
        src = src + src3
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
    
class decode_Adaptered(nn.Module):
    def __init__(self, orig_block,s):
        super().__init__()
        # self.adapter1 = Adapter()
        self.adapter2 = Adapter(s=s)
        # cross attention
        self.cross_attn = orig_block.cross_attn
        self.dropout1 =  orig_block.dropout1
        self.norm1 = orig_block.norm1

        # self attention
        self.self_attn = orig_block.self_attn
        self.dropout2 =  orig_block.dropout2
        self.norm2 = orig_block.norm2

        # ffn
        self.linear1 = orig_block.linear1
        self.activation = orig_block.activation
        self.dropout3 =  orig_block.dropout3
        self.linear2 = orig_block.linear2
        self.dropout4 =  orig_block.dropout4
        self.norm3 = orig_block.norm3

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt,novel=False):

        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        if novel:
            tgt3 = self.adapter2(self.dropout4(tgt2))
            tgt = tgt + tgt3
            tgt = self.norm3(tgt)

        else:
            tgt = tgt + self.dropout4(tgt2)
            tgt = self.norm3(tgt)

        return tgt
    
    def forward_ffn_text(self, tgt):
        tgt2 = self.linear_2_text(self.dropout3_text(self.activation_text(self.linear1_text(tgt))))
        tgt = tgt + self.dropout4_text(tgt2)
        tgt = self.norm3_text(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,novel=False, src_padding_mask=None):
        # self attention
        num_to_add = tgt.shape[1] - query_pos.shape[1]
        additional_query_pos = torch.randn(query_pos.shape[0], num_to_add, query_pos.shape[2], device=query_pos.device)
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
        # tgt3 = self.adapter1(tgt2)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        tgt= self.forward_ffn(tgt,novel)
        
        return tgt


class model_adapter(nn.Module):

    def __init__(self,model,kd,s=1.0):
        super().__init__()
        self.model = model
        self.kd = kd
        self.s = s
        # Freeze the original model parameters

        for name,params in self.model.named_parameters():
            if self.kd:
                if "class_embed_novel" not in name and "input_proj" not in name and "adapter" not in name and "novel_linear" not in name and "transformer.decoder.class_embed" not in name and "transformer.decoder.bbox_embed.6" not in name:
                    params.requires_grad = False
            else:
                if "class_embed" not in name and "input_proj" not in name:
                    params.requires_grad = False
                else:
                    params.requires_grad = True
        
        for i in range(6):
            self.model.transformer.decoder.layers[i] = decode_Adaptered(self.model.transformer.decoder.layers[i],s=self.s)

    def get_model(self):

        return self.model