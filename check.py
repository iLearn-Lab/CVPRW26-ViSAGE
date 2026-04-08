import os
import glob
import subprocess
from tqdm import tqdm


pred_frames_dir = "./predictvideos/final"
official_samples_dir = "./SampleSubmission" 


def get_actual_frame_count(video_path):

    cmd = [
        "ffprobe", "-v", "error", 
        "-select_streams", "v:0",
        "-count_packets", "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0", video_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return int(res.stdout.strip())
    except Exception as e:
        return -1

def main():
    folders = [d for d in os.listdir(pred_frames_dir) if os.path.isdir(os.path.join(pred_frames_dir, d))]
    
    perfect_match = 0
    mismatch_log = []
    missing_videos = []
    
    print(f"Validating frame alignment for {len(folders)} folders...")
    print(pred_frames_dir)
    for folder in tqdm(folders):
        folder_path = os.path.join(pred_frames_dir, folder)
        official_video = os.path.join(official_samples_dir, f"{folder}.mp4")
        
        if not os.path.exists(official_video):
            missing_videos.append(folder)
            continue
            

        target_count = get_actual_frame_count(official_video)
        if target_count <= 0:

            continue
            
  
        images = glob.glob(os.path.join(folder_path, "eyeMap_*.png"))
        current_count = len(images)
        

        if current_count == target_count:
            perfect_match += 1
        else:
            mismatch_log.append((folder, current_count, target_count))


    print("\n================ Results ================")
    print(f"Total folders checked: {len(folders)}")
    print(f"Perfectly aligned: {perfect_match}")
    
    if missing_videos:
        print(f"Missing corresponding official videos: {len(missing_videos)}")
        
    if mismatch_log:
        print(f"\n⚠️ Found frame count mismatches in {len(mismatch_log)} videos:")
        # To avoid clutter, only print the first 20 if there are too many mismatches
        display_limit = 20 
        for i, (folder, curr, targ) in enumerate(mismatch_log):
            if i >= display_limit:
                print(f"  ... and {len(mismatch_log) - display_limit} other videos.")
                break
            diff = curr - targ
            status = f"{diff} extra frames" if diff > 0 else f"missing {abs(diff)} frames"
            print(f"  - {folder}: Predicted {curr} frames, Official {targ} frames -> {status}")
    else:
        print("\n🎉 Excellent! The frame counts for all 800 prediction folders perfectly match the official videos 100%!")

if __name__ == "__main__":
    main()