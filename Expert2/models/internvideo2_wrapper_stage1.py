
import sys
import os
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


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

class InternVideo2Wrapper(nn.Module):
    def __init__(self, pretrained_path, num_frames=16, extract_layers=[11, 23, 35, 47], freeze_backbone=True):
        super().__init__()
        self.extract_layers = extract_layers
        self.patch_size = 14
        

        dummy_config = DummyVisionConfig(num_frames)
        self.vision_encoder = pretrain_internvideo2_6b_patch14_224(dummy_config)
        


        
        
        repo_root = './InternVideo/InternVideo2/multi_modality'
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root) 
            
        full_state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        

        if repo_root in sys.path:
            sys.path.remove(repo_root)


        if 'model' in full_state_dict:
            full_state_dict = full_state_dict['model']
        elif 'module' in full_state_dict:
            full_state_dict = full_state_dict['module']
            
        vision_state_dict = {}
        for k, v in full_state_dict.items():

            if k.startswith('vision_encoder.'):
                vision_state_dict[k.replace('vision_encoder.', '')] = v
            elif not k.startswith('text_encoder.') and not k.startswith('audio_encoder.'):
                vision_state_dict[k] = v


        interpolate_pos_embed_internvideo2(vision_state_dict, self.vision_encoder, orig_t_size=4)
        

        msg = self.vision_encoder.load_state_dict(vision_state_dict, strict=False)


        self.intermediate_features = []
        self._register_hooks()

        if freeze_backbone:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            
        else:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            lora_config = LoraConfig(
                r=16, lora_alpha=32, target_modules=["qkv", "proj", "fc1", "fc2"], 
                lora_dropout=0.05, bias="none"
            )
            self.vision_encoder = get_peft_model(self.vision_encoder, lora_config)
            

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple) and len(output) == 2:
            x, residual = output
            feat = x + residual if residual is not None else x
        else:
            feat = output
        self.intermediate_features.append(feat)

    def _register_hooks(self):
        for layer_idx in self.extract_layers:
            self.vision_encoder.blocks[layer_idx].register_forward_hook(self.hook_fn)

    def forward(self, x):
        self.intermediate_features.clear() 
        B, C, T, H, W = x.shape
        H_p, W_p = H // self.patch_size, W // self.patch_size
        
        _ = self.vision_encoder(x, use_image=False, x_vis_only=True)

        features = []
        for feat in self.intermediate_features:
            feat = feat[:, 1:, :] 
            Dim = feat.shape[-1]
            feat = feat.view(B, T, H_p, W_p, Dim)
            feat = feat.permute(0, 4, 1, 2, 3).contiguous()
            features.append(feat)
            
        return features