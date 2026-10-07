import os
import sys
import time
import bz2
import zipfile
import requests
from tqdm import tqdm

def download_file(url, target_path, desc=None, max_retries=25):
    if os.path.exists(target_path):
        print(f"Already exists: {target_path}")
        return True
    
    temp_path = target_path + ".tmp"
    
    # First get total size
    total_size = None
    try:
        head_resp = requests.head(url, allow_redirects=True, timeout=20)
        if head_resp.status_code == 200:
            total_size = int(head_resp.headers.get("content-length", 0))
    except Exception as e:
        print(f"Warning: Could not get Content-Length from HEAD ({e})")

    retries = 0
    while retries < max_retries:
        try:
            initial_pos = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
            
            if total_size and initial_pos >= total_size:
                print(f"\nDownload already complete for {temp_path}")
                break

            headers = {}
            if initial_pos > 0:
                headers["Range"] = f"bytes={initial_pos}-"
                print(f"\nResuming {desc or target_path} from byte {initial_pos} ({initial_pos/(1024*1024):.2f} MB)...")
            else:
                print(f"\nStarting download for {desc or target_path}...")

            with requests.get(url, headers=headers, stream=True, timeout=45) as resp:
                if resp.status_code == 416: # Range Not Satisfiable -> already downloaded
                    break
                
                if resp.status_code not in (200, 206):
                    raise RuntimeError(f"Unexpected status code {resp.status_code}")

                if total_size is None:
                    cl = resp.headers.get("content-length")
                    if cl:
                        total_size = int(cl) + (initial_pos if resp.status_code == 206 else 0)

                mode = "ab" if initial_pos > 0 and resp.status_code == 206 else "wb"
                if mode == "wb":
                    initial_pos = 0

                with open(temp_path, mode) as f, tqdm(
                    total=total_size,
                    initial=initial_pos,
                    unit="B",
                    unit_scale=True,
                    desc=desc or os.path.basename(target_path)
                ) as bar:
                    for chunk in resp.iter_content(chunk_size=512 * 1024):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            bar.update(len(chunk))
            
            # Check if file size matches
            cur_size = os.path.getsize(temp_path)
            if total_size and cur_size < total_size:
                print(f"\nIncomplete download ({cur_size}/{total_size}), retrying...")
                retries += 1
                time.sleep(2)
                continue

            break # Succeeded!

        except (requests.exceptions.RequestException, TimeoutError, ConnectionError, Exception) as err:
            retries += 1
            print(f"\n[Retry {retries}/{max_retries}] Download interrupted: {err}. Retrying in 3s...")
            time.sleep(3)

    if total_size and os.path.getsize(temp_path) < total_size:
        raise RuntimeError(f"Failed to fully download {url} after {max_retries} retries.")

    if os.path.exists(target_path):
        os.remove(target_path)
    os.rename(temp_path, target_path)
    print(f"Successfully downloaded to {target_path}")
    return True

def extract_zip(zip_path, extract_dir):
    print(f"\nExtracting {zip_path} into {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    print(f"Extracted {zip_path}")

def decompress_bz2(bz2_path, out_path):
    print(f"\nDecompressing {bz2_path} to {out_path}...")
    with bz2.BZ2File(bz2_path, "rb") as source, open(out_path, "wb") as dest:
        while True:
            buf = source.read(1024 * 1024)
            if not buf:
                break
            dest.write(buf)
    print(f"Decompressed to {out_path}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Landmark detection model
    face_detect_dir = os.path.join(base_dir, "Face_Detection")
    landmark_dat = os.path.join(face_detect_dir, "shape_predictor_68_face_landmarks.dat")
    landmark_bz2 = os.path.join(face_detect_dir, "shape_predictor_68_face_landmarks.dat.bz2")
    if not os.path.exists(landmark_dat):
        url = "https://raw.githubusercontent.com/davisking/dlib-models/master/shape_predictor_68_face_landmarks.dat.bz2"
        download_file(url, landmark_bz2, "Face Landmark Model (bz2)")
        decompress_bz2(landmark_bz2, landmark_dat)
        if os.path.exists(landmark_bz2):
            os.remove(landmark_bz2)
    else:
        print(f"Landmark model already present: {landmark_dat}")

    # 2. Face Enhancement checkpoints
    face_dir = os.path.join(base_dir, "Face_Enhancement")
    face_ckpt_zip = os.path.join(face_dir, "face_checkpoints.zip")
    face_ckpt_dir = os.path.join(face_dir, "checkpoints")
    if not os.path.exists(face_ckpt_dir):
        url = "https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/face_checkpoints.zip"
        download_file(url, face_ckpt_zip, "Face Checkpoints (zip)")
        extract_zip(face_ckpt_zip, face_dir)
        if os.path.exists(face_ckpt_zip):
            os.remove(face_ckpt_zip)
    else:
        print(f"Face checkpoints already present: {face_ckpt_dir}")

    # 3. Global checkpoints
    global_dir = os.path.join(base_dir, "Global")
    global_ckpt_zip = os.path.join(global_dir, "global_checkpoints.zip")
    global_ckpt_dir = os.path.join(global_dir, "checkpoints")
    if not os.path.exists(global_ckpt_dir):
        url = "https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip"
        download_file(url, global_ckpt_zip, "Global Checkpoints (zip)")
        extract_zip(global_ckpt_zip, global_dir)
        if os.path.exists(global_ckpt_zip):
            os.remove(global_ckpt_zip)
    else:
        print(f"Global checkpoints already present: {global_ckpt_dir}")

    print("\n--- Setup Models Completed Successfully ---")

if __name__ == "__main__":
    main()
