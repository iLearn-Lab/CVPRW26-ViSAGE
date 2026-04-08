
import sys
import os
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
import torch.utils.checkpoint as cp
from functools import partial


internvideo2_path = './InternVideo/InternVideo2/multi_modality/models'
if internvideo2_path not in sys.path:
    sys.path.insert(0, internvideo2_path)
from backbones.internvideo2.internvideo2 import pretrain_internvideo2_6b_patch14_224
from backbones.internvideo2.pos_embed import interpolate_pos_embed_internvideo2
if internvideo2_path in sys.path:
    sys.path.remove(internvideo2_path)


class VisionEncoderConfig:
    def __init__(self, num_frames):
        self.clip_embed_dim = 768
        self.use_flash_attn = True       
        self.use_fused_rmsnorm = True
        self.use_fused_mlp = True
        self.num_frames = num_frames     
        self.tubelet_size = 1
        self.sep_image_video_pos_embed = False
        self.use_checkpoint = False      
        self.checkpoint_num = 0          
        self.clip_teacher_embed_dim = 3200
        self.clip_teacher_final_dim = 768
        self.clip_norm_type = 'l2'
        self.clip_return_layer = 4
        self.clip_student_return_interval = 12
        self.pretrained = None 

    def get(self, key, default=None):
        return getattr(self, key, default)

class DummyVisionConfig:
    def __init__(self, num_frames=16):
        self.vision_encoder = VisionEncoderConfig(num_frames)


class FeatureExtractorBlock(nn.Module):
    def __init__(self, block, idx, use_cp, is_target, storage_dict):
        super().__init__()
        self.block = block
        self.idx = idx
        self.use_cp = use_cp
        self.is_target = is_target
        self.storage_dict = storage_dict
        
    def forward(self, *args, **kwargs):

        if self.use_cp:

            if kwargs:
                func = partial(self.block, **kwargs)
                out = cp.checkpoint(func, *args, use_reentrant=False)
            else:
                out = cp.checkpoint(self.block, *args, use_reentrant=False)
        else:
            out = self.block(*args, **kwargs)
            

        if self.is_target:
            if isinstance(out, tuple) and len(out) >= 2:
                feat_x, residual = out[0], out[1]
                feat = feat_x + residual if residual is not None else feat_x
            else:
                feat = out
            self.storage_dict[self.idx] = feat
            
        return out


class InternVideo2Wrapper(nn.Module):
    def __init__(self, pretrained_path, num_frames=16, extract_layers=[11, 23, 35, 47], freeze_backbone=True):
        super().__init__()
        self.extract_layers = extract_layers
        self.patch_size = 14
        
        
        dummy_config = DummyVisionConfig(num_frames)
        self.vision_encoder = pretrain_internvideo2_6b_patch14_224(dummy_config)
        

        repo_root = './InternVideo/InternVideo2/multi_modality'
        if repo_root not in sys.path: sys.path.insert(0, repo_root) 
        full_state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        if repo_root in sys.path: sys.path.remove(repo_root)

        if 'model' in full_state_dict: full_state_dict = full_state_dict['model']
        elif 'module' in full_state_dict: full_state_dict = full_state_dict['module']
            
        vision_state_dict = {}
        for k, v in full_state_dict.items():
            if k.startswith('vision_encoder.'): vision_state_dict[k.replace('vision_encoder.', '')] = v
            elif not k.startswith('text_encoder.') and not k.startswith('audio_encoder.'): vision_state_dict[k] = v

        interpolate_pos_embed_internvideo2(vision_state_dict, self.vision_encoder, orig_t_size=4)
        msg = self.vision_encoder.load_state_dict(vision_state_dict, strict=False)
        

        self.intermediate_dict = {}


        for i in range(len(self.vision_encoder.blocks)):
            use_cp = (i < 40) 
            is_target = (i in self.extract_layers)
            
            original_block = self.vision_encoder.blocks[i]

            self.vision_encoder.blocks[i] = FeatureExtractorBlock(
                block=original_block,
                idx=i,
                use_cp=use_cp,
                is_target=is_target,
                storage_dict=self.intermediate_dict
            )

        if freeze_backbone:
            for param in self.vision_encoder.parameters(): param.requires_grad = False
            
        else:
            for param in self.vision_encoder.parameters(): param.requires_grad = False
            
            lora_config = LoraConfig(
                r=64, lora_alpha=128, target_modules=["qkv", "proj", "fc1", "fc2"], 
                lora_dropout=0.05, bias="none"
            )
            self.vision_encoder = get_peft_model(self.vision_encoder, lora_config)
            

    def forward(self, x):
        self.intermediate_dict.clear() 
        B, C, T, H, W = x.shape
        H_p, W_p = H // self.patch_size, W // self.patch_size
        
        
        if x.is_floating_point():
            x.requires_grad_(True)
            
        _ = self.vision_encoder(x, use_image=False, x_vis_only=True)

        features = []
        for idx in self.extract_layers:
            feat = self.intermediate_dict[idx]
            feat = feat[:, 1:, :] 
            Dim = feat.shape[-1]
            feat = feat.view(B, T, H_p, W_p, Dim)
            feat = feat.permute(0, 4, 1, 2, 3).contiguous()
            features.append(feat)
            
        return features