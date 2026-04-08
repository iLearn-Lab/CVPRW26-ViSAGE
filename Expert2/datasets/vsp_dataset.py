# 文件路径: VSP_Project/datasets/vsp_dataset.py

import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ChallengeVSPDataset(Dataset):
    def __init__(self, cfg, split='train', custom_json=None, custom_root=None):

        self.cfg = cfg
        self.num_frames = cfg.NUM_FRAMES
        self.input_size = cfg.INPUT_SIZE

        self.offset_dict = {}
        offset_json_path = '/root/autodl-tmp/challenge/best_offsets.json' 
        
        if os.path.exists(offset_json_path):
            import json
            with open(offset_json_path, 'r') as f:
                self.offset_dict = json.load(f)
            
      
        json_path = custom_json if custom_json else cfg.JSON_PATH
        root_dir = custom_root if custom_root else cfg.ROOT_DIR
        
        with open(json_path, 'r') as f:
            splits_data = json.load(f)
            
        self.video_ids = splits_data.get(split, [])
        
        # 物理路径映射：如果是 val 也去 train 文件夹找
        physical_split = 'train' if split in ['train', 'val'] else split
        
        self.frames_base_dir = os.path.join(root_dir, 'frames', physical_split)
        self.gt_base_dir = os.path.join(root_dir, 'gt_maps', physical_split)
        # --------------------
        
        # 3. 定义预处理 Transform
        self.video_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=cfg.CLIP_MEAN, std=cfg.CLIP_STD)
        ])
        
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor() 
        ])

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid_raw = self.video_ids[idx]
        vid = vid_raw.replace('.mp4', '') 
        
        vid_frames_dir = os.path.join(self.frames_base_dir, vid)
        vid_gt_dir = os.path.join(self.gt_base_dir, vid)
        

        if not os.path.exists(vid_frames_dir):

            return torch.zeros((3, self.num_frames, self.input_size, self.input_size)), \
                   torch.zeros((1, self.num_frames, self.input_size, self.input_size)), \
                   torch.zeros((1, self.num_frames, self.input_size, self.input_size))

        all_frames = sorted([f for f in os.listdir(vid_frames_dir) if f.endswith('.jpg')])
        total_frames = len(all_frames)
        

        if total_frames > self.num_frames:
            import random
            start_idx = random.randint(0, total_frames - self.num_frames)
            indices = list(range(start_idx, start_idx + self.num_frames))
        elif total_frames == self.num_frames:
            indices = list(range(self.num_frames))
        else:
            indices = [i % total_frames for i in range(self.num_frames)]

        video_tensors = []
        gt_tensors = []
        fixation_masks = []
        

        try:
            sample_img = Image.open(os.path.join(vid_frames_dir, all_frames[indices[0]]))
            orig_w, orig_h = sample_img.size 
        except Exception:
            orig_w, orig_h = 1920, 1080 


        fix_json_path = os.path.join(self.cfg.FIXATION_DIR, vid, 'fixations.json')
        fix_data = None
        if os.path.exists(fix_json_path):
            try:
                with open(fix_json_path, 'r') as f:
                    fix_data = json.load(f)
            except Exception:
                fix_data = None

        vid_name = os.path.basename(vid_frames_dir) 
        

        current_offset = self.offset_dict.get(vid_name, 53)


        fps_ratio = 6 
        
        for i in indices:
            img_name = all_frames[i]
            idx_str = img_name.replace('img_', '').replace('.jpg', '')
            gt_name = f"eyeMap_{idx_str}.png"
            
            img_path = os.path.join(vid_frames_dir, img_name)
            gt_path = os.path.join(vid_gt_dir, gt_name)
            

            img = Image.open(img_path).convert('RGB')
            video_tensors.append(self.video_transform(img))
            

            if os.path.exists(gt_path):
                gt_img = Image.open(gt_path).convert('L')
                gt_tensors.append(self.gt_transform(gt_img))
            else:
                gt_tensors.append(torch.zeros((1, self.input_size, self.input_size)))

            f_mask = torch.zeros((1, self.input_size, self.input_size))
            
            if fix_data:
                orig_w, orig_h = img.size
                

                json_idx = i * 6 + 2
                
                
                if 0 <= json_idx < len(fix_data):
                    
                    points = fix_data[json_idx]
                    for pt in points:
                        if len(pt) >= 2: 
                            try:
                             
                                x_raw, y_raw = pt[1], pt[0]
                                
                                x_s = int(x_raw * self.input_size / orig_w)
                                y_s = int(y_raw * self.input_size / orig_h)
                                
                                if 0 <= y_s < self.input_size and 0 <= x_s < self.input_size:
                                    f_mask[0, y_s, x_s] += 1.0 
                            except (IndexError, TypeError): 
                                continue


                if f_mask.sum() > 0:
                    import torch.nn.functional as F
                    f_mask = F.max_pool2d(f_mask.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
            
            fixation_masks.append(f_mask)
                

        video_data = torch.stack(video_tensors, dim=0).permute(1, 0, 2, 3)
        gt_data = torch.stack(gt_tensors, dim=0).permute(1, 0, 2, 3)
        fixation_data = torch.stack(fixation_masks, dim=0).permute(1, 0, 2, 3)
        
        return video_data, gt_data, fixation_data