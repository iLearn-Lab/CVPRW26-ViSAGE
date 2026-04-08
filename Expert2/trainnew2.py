
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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
        

        self.offset_dict = {}
        offset_json_path = '/root/autodl-tmp/challenge/best_offsets.json' 
        if os.path.exists(offset_json_path):
            with open(offset_json_path, 'r') as f:
                self.offset_dict = json.load(f)
        
        for vid in val_videos:
            vid_name = vid.replace('.mp4', '')
            vid_frames_dir = os.path.join(cfg.VAL_DIR, 'frames', 'train', vid_name)
            vid_gt_dir = os.path.join(cfg.VAL_DIR, 'gt_maps', 'train', vid_name)

            fix_json_path = os.path.join(cfg.FIXATIONVAL_DIR, 'Train', vid_name, 'fixations.json')
            
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
        

        current_offset = self.offset_dict.get(vid_name, 53)
        
        v_tensors, gt_tensors, fix_tensors = [], [], [] 
        
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
        
        return video_tensor, gt_tensor, fixation_tensor, actual_len

def main():

    cfg = Config()
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(parent_dir, 'checkpoints_stage2topdown', f'run_{run_timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    
    train_dataset_26 = ChallengeVSPDataset(cfg=cfg, split='train', custom_root=cfg.ROOT_DIR)
    train_loader = DataLoader(
        train_dataset_26, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        num_workers=cfg.NUM_WORKERS, 
        pin_memory=True
    )
    
    json_path = 'TrainValSplit.json'
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
    
    # 解除主干封印，挂载 LoRA
    model = VSPModel(
        pretrained_path=pretrained_path,
        extract_layers=[11, 23, 35, 47], 
        freeze_backbone=False 
    )

    stage1_path = 'best.pth' # 帮您替换成了最新的路径
    
    if os.path.exists(stage1_path):

        stage1_weights = torch.load(stage1_path, map_location='cpu')
        
        model.load_state_dict(stage1_weights, strict=False)


    
    model = model.to(device)

    lora_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'encoder' in name:
            lora_params.append(param)
        else:
            decoder_params.append(param)


    lora_names = [n for n, p in model.named_parameters() if p.requires_grad and 'encoder' in n]


    optimizer = optim.AdamW([
        {'params': decoder_params, 'lr': 1e-5},
        {'params': lora_params, 'lr': 1e-4} 
    ], weight_decay=1e-4) 
    
    criterion = VSPLoss().to(device) 
    epochs = 20

    warmup_epochs = 2
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, total_iters=len(train_loader) * warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * (epochs - warmup_epochs), eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[len(train_loader) * warmup_epochs]
    )

    class ModelEMA:
        def __init__(self, model, decay=0.999):
            self.decay = decay
            self.shadow = {}
            self.backup = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.data.clone()
            for name, buffer in model.named_buffers():
                self.shadow[name] = buffer.data.clone()

        def update(self, model):
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        self.shadow[name].lerp_(param.data, 1.0 - self.decay)
                for name, buffer in model.named_buffers():
                    self.shadow[name].copy_(buffer.data)

        def apply_shadow(self, model):
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.backup[name] = param.data.clone()
                    param.data = self.shadow[name].clone()
            for name, buffer in model.named_buffers():
                self.backup[name] = buffer.data.clone()
                buffer.data = self.shadow[name].clone()

        def restore(self, model):
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data = self.backup[name].clone()
            for name, buffer in model.named_buffers():
                buffer.data = self.backup[name].clone()
            self.backup = {}


    ema = ModelEMA(model, decay=0.99)
    experiment_info = {
        "timestamp": run_timestamp,
        "mode": "Stage 2 (LoRA Full Finetuning)",
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": cfg.BATCH_SIZE,
            "lr_decoder": 1e-5,
            "lr_lora": 1e-4,
            "weight_decay": 1e-4,
            "validation_strategy": "4-Crop Full Validation" 
        },
        "model_architecture": {
            "backbone": "InternVideo2-6B",
            "frozen_backbone": False,
            "lora_rank": 16,
            "stage1_weights": stage1_path 
        }
    }

    info_path = os.path.join(run_dir, 'experiment_config.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_info, f, indent=4, ensure_ascii=False)
    
  
    best_cc = -1.0 
    top_k_models = []  
    MAX_SAVE = 4       

    for epoch in range(epochs):

        model.train()
        for m in model.decoder.modules():
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm3d) or isinstance(m, torch.nn.BatchNorm1d):
                m.eval()
        total_loss = 0.0
        
        accumulation_steps = 1  
        optimizer.zero_grad() 
        
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs} [Train]")
        

        for i, (videos, gts, fixations) in train_bar:
            videos = videos.to(device).bfloat16()
            gts = gts.to(device).float()
            fixations = fixations.to(device).float()
            
  
            preds = model(videos)
            loss = criterion(preds.float(), gts, fixations=fixations) 
            loss = loss / accumulation_steps
            loss.backward()
            
            if ((i + 1) % accumulation_steps == 0) or ((i + 1) == len(train_loader)):
                optimizer.step()
                scheduler.step()   
                ema.update(model)  
                optimizer.zero_grad() 
            
            current_step_loss = loss.item() * accumulation_steps
            total_loss += current_step_loss
            train_bar.set_postfix({'Loss': f"{current_step_loss:.4f}"})
            
        avg_train_loss = total_loss / len(train_loader)
        

        model.eval()
        ema.apply_shadow(model)
        total_cc_sum = 0.0
        total_sim_sum = 0.0
        total_nss_sum = 0.0  
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

                    total_cc_sum += chunk_cc * a_len
                    total_sim_sum += chunk_sim * a_len
                    total_nss_sum += chunk_nss.item() * a_len 
                    total_valid_frames += a_len
                    
        if total_valid_frames > 0:
            avg_cc = total_cc_sum / total_valid_frames
            avg_sim = total_sim_sum / total_valid_frames
            avg_nss = total_nss_sum / total_valid_frames 
        else:
            avg_cc, avg_sim, avg_nss = 0.0, 0.0, 0.0
        

        is_top_k = False
        if len(top_k_models) < MAX_SAVE:
            is_top_k = True
        elif avg_cc > top_k_models[-1]['val_cc']: 
            is_top_k = True

        if is_top_k:
  
            model_name = f"vsp_stage2_epoch{epoch+1}_cc{avg_cc:.4f}.pth"
            current_model_path = os.path.join(run_dir, model_name)
            
            full_state = model.state_dict()
            stage2_state_dict = {}
            
            trainable_names = [k for k, p in model.named_parameters() if p.requires_grad]
            for k in trainable_names:
                if k in full_state:
                    stage2_state_dict[k] = full_state[k].cpu()
                    
            for k, v in full_state.items():
                if k.startswith('decoder.') or k.startswith('dim_reducers.') or k.startswith('film_'):
                    if k not in stage2_state_dict:
                        stage2_state_dict[k] = v.cpu()
            

            torch.save(stage2_state_dict, current_model_path)
            

            top_k_models.append({
                "epoch": epoch + 1,
                "val_cc": float(avg_cc),
                "val_sim": float(avg_sim),
                "val_nss": float(avg_nss),
                "train_loss": float(avg_train_loss),
                "path": current_model_path
            })
            

            top_k_models = sorted(top_k_models, key=lambda x: x['val_cc'], reverse=True)
            

            if len(top_k_models) > MAX_SAVE:
                worst_model = top_k_models.pop() 
                if os.path.exists(worst_model['path']):
                    os.remove(worst_model['path']) 
                    

            metrics_path = os.path.join(run_dir, "best_metrics.json")
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump({"top_models": top_k_models}, f, indent=4, ensure_ascii=False)
                
            
        else:
            print("\n")
            
        ema.restore(model)

if __name__ == '__main__':
    main()