import subprocess
from pathlib import Path
from tqdm import tqdm


PRED_DIR = Path("./predictvideos/final")

MP4_DIR  = Path("./SampleSubmission")

OUT_DIR  = Path("./predictvideos/submission")
# ============================================

def get_exact_fps(mp4_path: Path) -> str:

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(mp4_path)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        fps = res.stdout.strip()
        return fps if fps else "30/1"
    except Exception:
        return "30/1" 

def main():

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    

    folders = [d for d in PRED_DIR.iterdir() if d.is_dir()]
    


    error_list = []

    for folder in tqdm(folders, desc="progress"):
        vid = folder.name
        original_mp4 = MP4_DIR / f"{vid}.mp4"
        out_mp4 = OUT_DIR / f"{vid}.mp4"
        

        fps = get_exact_fps(original_mp4)
        

        img_pattern = str(folder / "eyeMap_%05d.png")

        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-framerate", fps,
            "-i", img_pattern,
            "-c:v", "libx264",    
            "-pix_fmt", "yuv420p",
            str(out_mp4)
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            error_list.append(vid)
            

    print("\n================ Done ================")
    if error_list:
        print(f"⚠️ Failed to synthesize {len(error_list)} videos. The list is as follows:")
        for err in error_list:
            print(f"  - {err}")
    else:
        print(f"🎉 Success! All 800 videos have been synthesized according to official standards.")
        print(f"📁 Your final submission videos are located at: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()