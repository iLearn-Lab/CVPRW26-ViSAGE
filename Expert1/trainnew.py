
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import json
from datetime import datetime

from configs.default_config import Config
from datasets.vsp_dataset import ChallengeVSPDataset
from models.full_model import VSPModel
from utils.losses import VSPLoss, torch_nss
from utils.metrics import calc_cc_sim_batch

class MultiCropValidationDataset(Dataset):
    def __init__(self, cfg, val_videos, num_clips=4):
        self.cfg = cfg
        self.window_size = cfg.NUM_FRAMES
        self.num_clips = num_clips
        self.chunks = []
        
        self.video_transform = transforms.Compose([
            transforms.Resize((cfg.INPUT_SIZE, cfg.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.CLIP_MEAN, std=cfg.CLIP_STD)
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((cfg.INPUT_SIZE, cfg.INPUT_SIZE)),
            transforms.ToTensor() 
        ])

        for vid in val_videos:

            vid_name=vid
            vid_frames_dir = os.path.join(cfg.VAL_DIR, 'frames', 'train', vid_name)
            vid_gt_dir = os.path.join(cfg.VAL_DIR, 'gt_maps', 'train', vid_name)

            fix_json_path = os.path.join(cfg.FIXATIONVAL_DIR,'Train', vid_name, 'fixations.json')
            
            if not os.path.exists(vid_frames_dir):
                continue
                
            all_frames = sorted([f for f in os.listdir(vid_frames_dir) if f.endswith('.jpg')])
            total_frames = len(all_frames)
            
            if total_frames == 0:
                continue
            
            if total_frames <= self.window_size:
                start_indices = [0]
            else:
                interval = (total_frames - self.window_size) / (self.num_clips - 1)
                start_indices = [int(i * interval) for i in range(self.num_clips)]
                
            for start_idx in start_indices:
                chunk_files = all_frames[start_idx : start_idx + self.window_size]
                self.chunks.append({
                    'chunk_files': chunk_files,
                    'frames_dir': vid_frames_dir,
                    'gt_dir': vid_gt_dir,
                    'fix_json_path': fix_json_path 
                })

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        chunk_files = chunk['chunk_files']
        actual_len = len(chunk_files)
        vid_name = os.path.basename(chunk['frames_dir'])
        

        
        
        v_tensors, gt_tensors, fix_tensors = [], [], [] # 🚨 增加 fix_tensors 列表

        try:
            sample_img = Image.open(os.path.join(chunk['frames_dir'], chunk_files[0]))
            orig_w, orig_h = sample_img.size
        except:
            orig_w, orig_h = 1920, 1080


        fix_data = None
        if os.path.exists(chunk['fix_json_path']):
            with open(chunk['fix_json_path'], 'r') as f:
                fix_data = json.load(f)

        for fname in chunk_files:
            img = Image.open(os.path.join(chunk['frames_dir'], fname)).convert('RGB')
            v_tensors.append(self.video_transform(img))
            
            gt_name = fname.replace('img_', 'eyeMap_').replace('.jpg', '.png')
            gt_path = os.path.join(chunk['gt_dir'], gt_name)
            if os.path.exists(gt_path):
                gt_img = Image.open(gt_path).convert('L')
                gt_tensors.append(self.gt_transform(gt_img))
            else:
                gt_tensors.append(torch.zeros((1, self.cfg.INPUT_SIZE, self.cfg.INPUT_SIZE)))

            f_mask = torch.zeros((1, self.cfg.INPUT_SIZE, self.cfg.INPUT_SIZE))
            
            if fix_data:
                orig_w, orig_h = img.size 
                

                frame_idx_5fps = int(fname.replace('img_', '').replace('.jpg', '')) - 1
                json_idx = frame_idx_5fps * 6 + 2 
                

                json_idx = max(0, min(len(fix_data) - 1, json_idx))
                
                if json_idx < len(fix_data):

                    points = fix_data[json_idx] 
                    
                    for pt in points:
                        if len(pt) >= 2: 
                            try:
                               
                                x_raw, y_raw = pt[1], pt[0] 
                                
                               
                                x_s = int(x_raw * self.cfg.INPUT_SIZE / orig_w)
                                y_s = int(y_raw * self.cfg.INPUT_SIZE / orig_h)
                                
                                if 0 <= y_s < self.cfg.INPUT_SIZE and 0 <= x_s < self.cfg.INPUT_SIZE:
                                    
                                    f_mask[0, y_s, x_s] += 1.0
                            except (IndexError, TypeError):
                                continue


            if f_mask.sum() > 0:
                import torch.nn.functional as F
                
                f_mask = F.max_pool2d(f_mask.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)

            fix_tensors.append(f_mask)
        
        pad_len = self.window_size - actual_len
        if pad_len > 0:
            v_tensors.extend([v_tensors[-1]] * pad_len)
            gt_tensors.extend([gt_tensors[-1]] * pad_len)
            
        video_tensor = torch.stack(v_tensors, dim=0).permute(1, 0, 2, 3)
        gt_tensor = torch.stack(gt_tensors, dim=0).permute(1, 0, 2, 3)
        fixation_tensor = torch.stack(fix_tensors, dim=0).permute(1, 0, 2, 3)
        
        return video_tensor, gt_tensor, fixation_tensor,actual_len



def main():
    print("🚀 start.")
    cfg = Config()
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(parent_dir, 'checkpointsclaude2', f'run_{run_timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    
    train_dataset_26 = ChallengeVSPDataset(cfg=cfg, split='train', custom_root=cfg.ROOT_DIR)
    train_loader = DataLoader(
        train_dataset_26, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        num_workers=cfg.NUM_WORKERS, 
        pin_memory=True
    )
    
    json_path = 'TrainValSplitnew.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    val_videos = data.get('val', [])
    
    val_dataset = MultiCropValidationDataset(cfg=cfg, val_videos=val_videos, num_clips=4)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False, 
        num_workers=cfg.NUM_WORKERS, 
        pin_memory=True
    )
    

    

    pretrained_path = 'internvideo2-s2_6b-224p-f4_with_audio_encoder.pt'
    
   
    model = VSPModel(
        pretrained_path=pretrained_path,
        extract_layers=[11, 23, 35, 47], 
        freeze_backbone=True
    )
    

    model = model.to(device)
    
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    
    for name in trainable_names:
        print(f"  - {name}")
    print("========================================================\n")
    lr = 1e-4
    weight_decay = 1e-4
    epochs = 20
    
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    criterion = VSPLoss().to(device) 
    

    experiment_info = {
        "timestamp": run_timestamp,
        "mode": "Stage 1 (Decoder Only)",
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": cfg.BATCH_SIZE,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "num_frames": getattr(cfg, 'NUM_FRAMES', 16),
            "input_size": getattr(cfg, 'INPUT_SIZE', 224),
            "loss_type": "VSPLoss (KL, CC, SIM, BCE, NSS)", 
            "nss_weight": 1.0, 
            "validation_strategy": "4-Crop Uniform Sampling" 
        },
        "datasets": {
            "train_data": {
                "name": "2026 Challenge Saliency",
                "split": "train",
                "json_path": json_path,
                "root_dir": cfg.VAL_DIR 
            },
            "val_data": {
                "name": "2026 Challenge Saliency",
                "split": "val",
                "json_path": json_path,
                "root_dir": cfg.VAL_DIR 
            }
        },
        "model_architecture": {
            "backbone": "InternVideo2-6B", 
            "frozen_backbone": True
        }
    }

    info_path = os.path.join(run_dir, 'experiment_config.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_info, f, indent=4, ensure_ascii=False)
    print(f"📝 saved: {info_path}")
    

    best_cc = -1.0 
    

    for epoch in range(epochs):
        
        model.train()
        total_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for videos, gts, fixations in train_bar:
            videos = videos.to(device).bfloat16()
            gts = gts.to(device).float()
            fixations = fixations.to(device).float() 
            
            optimizer.zero_grad()
            preds = model(videos)
            loss = criterion(preds.float(), gts, fixations=fixations)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        avg_train_loss = total_loss / len(train_loader)
        

        model.eval()
        total_nss_sum = 0.0
        total_cc_sum = 0.0
        total_sim_sum = 0.0
        total_valid_frames = 0
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ")
        
        with torch.no_grad():
            for videos, gts, fixations, actual_lens in val_bar:
                videos = videos.to(device).bfloat16()
                gts = gts.to(device).float()
                fixations = fixations.to(device).float()
                
                preds = model(videos)
                
                for b in range(videos.size(0)):
                    a_len = actual_lens[b].item()
                    
                    pred_real = preds[b:b+1, :, :a_len, :, :]
                    gt_real = gts[b:b+1, :, :a_len, :, :]
                    f_real = fixations[b:b+1, :, :a_len, :, :] 
                    
                    chunk_cc, chunk_sim = calc_cc_sim_batch(pred_real.float(), gt_real.float())
                    chunk_nss = torch_nss(pred_real.float(), f_real.float())

                    total_nss_sum += chunk_nss.item() * a_len
                    total_cc_sum += chunk_cc * a_len
                    total_sim_sum += chunk_sim * a_len
                    total_valid_frames += a_len

                    active_pts = (fixations[b] > 0).sum().item()
                    if active_pts < 5: 

                    
        if total_valid_frames > 0:
            avg_cc = total_cc_sum / total_valid_frames
            avg_sim = total_sim_sum / total_valid_frames
            avg_nss = total_nss_sum / total_valid_frames 
        else:
            avg_cc, avg_sim = 0.0, 0.0
        
        
        if avg_cc > best_cc:
            best_cc = avg_cc
            best_model_path = os.path.join(run_dir, "cvmm_stage1_best.pth")
            
            
            full_state = model.state_dict()
            stage1_state_dict = {}
            for k, v in full_state.items():
                
                if k.startswith('decoder.') or k.startswith('dim_reducers.') or k.startswith('film_'):
                    stage1_state_dict[k] = v.cpu()
                    
            torch.save(stage1_state_dict, best_model_path)
            
            
            best_metrics = {
                "best_epoch": epoch + 1,
                "val_cc": float(best_cc),
                "val_sim": float(avg_sim),
                "val_nss": float(avg_nss), 
                "train_loss": float(avg_train_loss)
            }
            metrics_path = os.path.join(run_dir, "best_metrics.json")
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(best_metrics, f, indent=4, ensure_ascii=False)
                
            print(f"⭐ best {best_cc:.4f}，save {best_model_path}")
            print(f"📊 : {metrics_path}\n")
        else:
            print("\n")

if __name__ == '__main__':
    main()