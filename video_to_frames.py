import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(__file__).resolve().parent

SPLIT_JSON = BASE_DIR / "TrainTestSplit.json"
VIDEO_DIR  = BASE_DIR / "videos" / "all_video"  # Fill in the path to the extracted Video.zip
OUT_ROOT   = BASE_DIR / "derived_fullfps"       
NUM_WORKERS = 8

FRAMES_OUT = OUT_ROOT / "frames" / "test"
AUDIO_OUT  = OUT_ROOT / "audio" / "test"

def run(cmd):
    subprocess.run(cmd, check=True)

def has_audio_stream(mp4_path: Path) -> bool:

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(mp4_path)
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode == 0 and ("audio" in (p.stdout or "").strip())

def get_duration_sec(mp4_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(mp4_path)
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return float((p.stdout or "0").strip() or 0.0)

def extract_frames(mp4_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    if any(out_dir.glob("img_*.jpg")):
        return "skip_frames"

    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(mp4_path),
        "-q:v", "2",  
        str(out_dir / "img_%05d.jpg")
    ]
    run(cmd)
    return "ok_frames"

def extract_audio_or_silence(mp4_path: Path, out_wav: Path):
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if out_wav.exists() and out_wav.stat().st_size > 1024:
        return "skip_audio"

    if has_audio_stream(mp4_path):

        cmd = [
            "ffmpeg", "-v", "error",
            "-i", str(mp4_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            str(out_wav)
        ]
        run(cmd)
        return "ok_audio"
    else:

        dur = get_duration_sec(mp4_path)
        if dur <= 0:

            dur = 1.0
        cmd = [
            "ffmpeg", "-v", "error",
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=mono:sample_rate=16000",
            "-t", f"{dur:.6f}",
            str(out_wav)
        ]
        run(cmd)
        return "silence"

def process_one(mp4_name: str):
    mp4_path = VIDEO_DIR / mp4_name
    vid = Path(mp4_name).stem

    if not mp4_path.exists():
        return ("missing", vid, f"mp4 not found: {mp4_path}")


    frames_dir = FRAMES_OUT / vid
    wav_path   = AUDIO_OUT / vid / f"{vid}.wav"

    r1 = extract_frames(mp4_path, frames_dir)
    r2 = extract_audio_or_silence(mp4_path, wav_path)
    return ("ok", vid, (r1, r2))

def main():
    split = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))


    public_key  = "public_test"  if "public_test"  in split else ("public"  if "public"  in split else None)
    private_key = "private_test" if "private_test" in split else ("private" if "private" in split else None)
    if public_key is None or private_key is None:
        raise KeyError(f"Split json keys not found. Got keys: {list(split.keys())}")

    mp4_list = list(split[public_key]) + list(split[private_key])

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FRAMES_OUT.mkdir(parents=True, exist_ok=True)
    AUDIO_OUT.mkdir(parents=True, exist_ok=True)

    stats = {"ok":0, "missing":0, "fail":0, "silence":0, "skip_frames":0, "skip_audio":0}
    print(f"Total test videos: {len(mp4_list)} (public={len(split[public_key])}, private={len(split[private_key])})")
    print(f"VIDEO_DIR: {VIDEO_DIR.resolve()}")
    print(f"FRAMES  : {FRAMES_OUT.resolve()}")
    print(f"AUDIO   : {AUDIO_OUT.resolve()}")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = [ex.submit(process_one, mp4_name) for mp4_name in mp4_list]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                status, vid, info = fut.result()
                if status == "missing":
                    stats["missing"] += 1
                    print(f"[MISS] {vid}: {info}")
                else:
                    stats["ok"] += 1
                    r1, r2 = info
                    if r1 == "skip_frames": stats["skip_frames"] += 1
                    if r2 == "skip_audio":  stats["skip_audio"]  += 1
                    if r2 == "silence":     stats["silence"]     += 1
            except Exception as e:
                stats["fail"] += 1
                print(f"[FAIL] {e}")

            if i % 20 == 0:
                print(f"Progress {i}/{len(futs)} | ok={stats['ok']} miss={stats['missing']} fail={stats['fail']} silence={stats['silence']}")

    print("Done:", stats)

if __name__ == "__main__":
    main()
