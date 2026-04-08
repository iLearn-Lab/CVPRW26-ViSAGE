

import os
import sys
import json
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from configs.default_config import Config
from models.full_model import VSPModel

def main():
    cfg = Config()
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    with open(cfg.JSON_PATH, 'r') as f:
        data = json.load(f)
    
    test_splits = {
        'public_test': data.get('public_test', data.get('public test', [])),
        'private_test': data.get('private_test', data.get('private test', []))
    }
    
    total_test_videos = len(test_splits['public_test']) + len(test_splits['private_test'])
    if total_test_videos == 0:
        return



    pretrained_path = 'yourpath/internvideo2-s2_6b-224p-f4_with_audio_encoder.pt'#Please insert the path to the downloaded pre-trained weights here.
    
    model = VSPModel(
        pretrained_path=pretrained_path,
        extract_layers=[11, 23, 35, 47],
        freeze_backbone=False  
    )
    

    model.encoder.vision_encoder.num_frames = getattr(cfg, 'NUM_FRAMES', 20)
    model = model.to(device)
    

    ckpt_path = './checkpoints/expert1.pth'

    
    if os.path.exists(ckpt_path):

        state_dict = torch.load(ckpt_path, map_location=device)
        

        has_backbone = any('encoder' in k or 'backbone' in k for k in state_dict.keys())
        


        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        lora_in_unexpected = [k for k in unexpected_keys if 'lora' in k.lower()]

    
    model.eval() 


    transform = transforms.Compose([
        transforms.Resize((cfg.INPUT_SIZE, cfg.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.CLIP_MEAN, std=cfg.CLIP_STD)
    ])
    to_pil = transforms.ToPILImage()


    current_dir = os.path.dirname(os.path.abspath(__file__))

    project_root = os.path.dirname(current_dir)

    output_base_dir = os.path.join(project_root, 'prediction', 'expert1')

    os.makedirs(output_base_dir, exist_ok=True)

    window_size = getattr(cfg, 'NUM_FRAMES', 20) 
    
    TEST_FPS = 30
    TRAIN_FPS = 5
    fps_stride = TEST_FPS // TRAIN_FPS  
    

    
    with torch.no_grad():
        for split_name, test_videos in test_splits.items():
            if not test_videos:
                continue
                
            for vid in tqdm(test_videos, desc=f"Inference [{split_name}]"):
                vid_name = vid.replace('.mp4', '')
                

                vid_frames_dir = os.path.join(cfg.TEST_DIR, 'frames', 'test', vid_name)
                
                if not os.path.exists(vid_frames_dir):

                    continue

                vid_out_dir = os.path.join(output_base_dir, vid_name)
                os.makedirs(vid_out_dir, exist_ok=True)

                all_frames = sorted([f for f in os.listdir(vid_frames_dir) if f.endswith('.jpg')])
                total_frames = len(all_frames)
                if total_frames == 0:
                    continue
                

                for offset in range(fps_stride):
                    sub_frames = all_frames[offset::fps_stride]
                    if not sub_frames:
                        continue
                        
                    sub_total = len(sub_frames)

                    for i in range(0, sub_total, window_size):
                        chunk_files = sub_frames[i : i + window_size]

                        all_exist = True
                        for _fname in chunk_files:
                            _out_fname = _fname.replace('img_', 'eyeMap_').replace('.jpg', '.png')
                            if not os.path.exists(os.path.join(vid_out_dir, _out_fname)):
                                all_exist = False
                                break

                        if all_exist:
                            continue
                        actual_len = len(chunk_files)
                        
                        tensors = []
                        for fname in chunk_files:
                            img_path = os.path.join(vid_frames_dir, fname)
                            img = Image.open(img_path).convert('RGB')
                            tensors.append(transform(img))
                        

                        pad_len = window_size - actual_len
                        if pad_len > 0:
                            tensors.extend([tensors[-1]] * pad_len)
                            
                        video_tensor = torch.stack(tensors, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
                        video_tensor = video_tensor.to(device)
                        
                        pred_saliency = model(video_tensor) 
                        

                        pred_saliency = pred_saliency[0, 0, :actual_len] 
                        
                        for j in range(actual_len):
                            orig_fname = chunk_files[j]
                            out_fname = orig_fname.replace('img_', 'eyeMap_').replace('.jpg', '.png')
                            
                            pred_frame = pred_saliency[j].cpu().float() 
                            pred_img = to_pil(pred_frame.unsqueeze(0)) 
                            pred_img.save(os.path.join(vid_out_dir, out_fname))

    print(f"\n✅ Done: {output_base_dir}")

if __name__ == '__main__':
    main()