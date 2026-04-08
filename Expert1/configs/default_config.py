

import os

class Config:

    ROOT_DIR = '/root/autodl-tmp/derived_5fps'

    # VAL_DIR = '/root/autodl-tmp/derived_5fps'
    VAL_DIR = '/root/autodl-tmp/challenge26/2026derived_5fps_blurred_train'

    FIXATION_DIR = '/root/autodl-tmp/FixationTrain1' 
    FIXATIONVAL_DIR = '/root/autodl-tmp/challenge26/FixationTrain'
    TEST_DIR = './derived_fullfps'
    # JSON_PATH = '/root/lizhiran/challenge/val_videos.json'
    JSON_PATH = './TrainTestSplit.json'

    ORIGINAL_FPS = 30           
    
    NUM_FRAMES = 20         
    INPUT_SIZE = 224        
    BATCH_SIZE = 4         
    NUM_WORKERS = 4         
    

    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

