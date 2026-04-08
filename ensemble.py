import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_gray_as_float(path):

    img = np.array(Image.open(path).convert('L'), dtype=np.float64) / 255.0
    return img

def to_logit(p, eps=1e-6):

    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def to_prob(logit):

    return 1.0 / (1.0 + np.exp(-logit))

def ensemble_frame(img1, img2, alpha=0.5, mode='logit'):

    if mode == 'mean':
        return alpha * img1 + (1 - alpha) * img2

    elif mode == 'logit':
        logit1 = to_logit(img1)
        logit2 = to_logit(img2)
        logit_fused = alpha * logit1 + (1 - alpha) * logit2
        return to_prob(logit_fused)

    elif mode == 'max':
        return np.maximum(img1, img2)

    elif mode == 'hybrid':
        mean_result = alpha * img1 + (1 - alpha) * img2
        max_result = np.maximum(img1, img2)

        threshold = 0.4
        mask = (mean_result > threshold).astype(np.float64)
        return mask * max_result + (1 - mask) * mean_result

    else:
        raise ValueError(f"Unknown mode: {mode}")

def ensemble_predictions(dir1, dir2, output_dir, alpha=0.5, mode='logit'):
    print(f"🚀 mode: {mode} |  alpha={alpha:.2f}")
    os.makedirs(output_dir, exist_ok=True)

    val_videos = sorted([d for d in os.listdir(dir1) if os.path.isdir(os.path.join(dir1, d))])
    if not val_videos:

        return

    for vid in tqdm(val_videos, desc="Ensemble Progress"):
        vid_dir1 = os.path.join(dir1, vid)
        vid_dir2 = os.path.join(dir2, vid)
        out_vid_dir = os.path.join(output_dir, vid)

        if not os.path.exists(vid_dir2):
            continue
        os.makedirs(out_vid_dir, exist_ok=True)

        frames = sorted([f for f in os.listdir(vid_dir1) if f.endswith(('.png', '.jpg'))])

        for frame in frames:
            path1 = os.path.join(vid_dir1, frame)
            path2 = os.path.join(vid_dir2, frame)
            if not os.path.exists(path2):
                continue

            img1 = load_gray_as_float(path1)
            img2 = load_gray_as_float(path2)

            blended = ensemble_frame(img1, img2, alpha=alpha, mode=mode)
            blended_uint8 = np.clip(blended * 255, 0, 255).astype(np.uint8)

            Image.fromarray(blended_uint8).save(os.path.join(out_vid_dir, frame))

    print(f"✅ done → {output_dir}")

if __name__ == "__main__":
    DIR_1 = "./prediction/expert1"
    DIR_2 = "./prediction/expert2"



    ensemble_predictions(DIR_1, DIR_2,
        output_dir = "./predictvideos/final",
        alpha=0.5, mode='logit')

