

import os

class Config:

    ROOT_DIR = '/root/autodl-tmp/derived_5fps'

    # VAL_DIR = '/root/autodl-tmp/derived_5fps'
    VAL_DIR = '/root/autodl-tmp/challenge26/2026derived_5fps_blurred'

    FIXATION_DIR = '/root/autodl-tmp/FixationTrain1' # 确保该目录下有各视频ID的子文件夹
    FIXATIONVAL_DIR = '/root/autodl-tmp/challenge26/FixationTrain'
    TEST_DIR = './derived_fullfps'
    # JSON_PATH = '/root/lizhiran/challenge/val_videos.json'
    JSON_PATH = './TrainTestSplit.json'
    # 
    # 🚨 确定性的对齐参数
    ORIGINAL_FPS = 30           
    
    # --- 数据超参数 ---
    NUM_FRAMES = 20         # 每次输入的视频帧数
    INPUT_SIZE = 224        # InternVideo 默认的空间分辨率
    BATCH_SIZE = 4         # 根据你的显存大小调整（如果 OOM 就调成 2 或 1）
    NUM_WORKERS = 4         # DataLoader 读取数据的多线程数
    
    # --- 预处理参数 (InternVideo/CLIP 默认) ---
    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

    # --- 模型路径 ---
    # 请把你下载好的 internvideo_next 权重路径填在这里
    # MODEL_WEIGHTS = '/root/lizhiran/models/internvideo_next_large.pth'