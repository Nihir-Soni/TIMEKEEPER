import os
import shutil
import base64
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Import the existing pipeline logic from GUI.py
from GUI import modify, make_preview_bytes

app = FastAPI(title="TIMEKEEPER Restoration API")

# Allow CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "output")
TEMP_INPUT_FOLDER = os.path.join(PROJECT_ROOT, "gui_temp_input")

def encode_image(img_path):
    """Read an image and return base64 encoded string"""
    if not os.path.exists(img_path):
        return None
    img = cv2.imread(img_path)
    if img is None:
        return None
    _, buffer = cv2.imencode('.png', img)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

def encode_bytes(img_bytes):
    """Return base64 string from raw png bytes"""
    if not img_bytes:
        return None
    return f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"

@app.post("/restore")
async def restore_photo(
    file: UploadFile = File(...),
    scratchRemoval: bool = Form(False),
    colorization: bool = Form(True),
    highResolution: bool = Form(False),
    uncertainty: bool = Form(True),
    saturation: float = Form(1.0)
):
    try:
        # 1. Setup temp folder
        if os.path.exists(TEMP_INPUT_FOLDER):
            shutil.rmtree(TEMP_INPUT_FOLDER)
        os.makedirs(TEMP_INPUT_FOLDER, exist_ok=True)
        
        # 2. Save uploaded file
        file_path = os.path.join(TEMP_INPUT_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 3. Call the modify function
        def status_callback(msg):
            print(f"[API Status] {msg}")
            
        modify(
            image_filename=TEMP_INPUT_FOLDER,
            with_scratch=scratchRemoval,
            with_colorize=colorization,
            with_hr=highResolution,
            status_callback=status_callback,
            saturation=saturation
        )
        
        # 4. Gather results
        base_name = file.filename
        base_name_no_ext = os.path.splitext(base_name)[0]
        final_dir = os.path.join(OUTPUT_FOLDER, "final_output")
        
        candidates = [
            os.path.join(final_dir, f"{base_name_no_ext}.png"),
            os.path.join(final_dir, base_name),
        ]
        
        out_img_path = None
        for cand in candidates:
            if os.path.exists(cand):
                out_img_path = cand
                break
                
        if not out_img_path:
            return {"error": "Processing failed, output image not found."}
            
        # Result dictionary
        results = {
            "original": encode_image(file_path),
            "restored": encode_image(out_img_path) if scratchRemoval else None,
            "colorized": None,
            "uncertainty": None,
            "uncertainty_reason": None,
            "confidence": None,
            "confidence_reason": None
        }
        
        if colorization:
            # The final image is colorized if colorization was selected
            results["colorized"] = encode_image(out_img_path)
            
            # Create a grayscale version for the 'restored' output ONLY if scratch removal was also selected
            if scratchRemoval:
                out_img_cv = cv2.imread(out_img_path)
                if out_img_cv is not None:
                    gray_img = cv2.cvtColor(out_img_cv, cv2.COLOR_BGR2GRAY)
                    gray_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
                    _, buffer = cv2.imencode('.png', gray_bgr)
                    results["restored"] = f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"
            
        # 5. Handle Uncertainty Map
        if uncertainty:
            try:
                from research.uncertainty import UncertaintyInferencer, UncertaintyUNet
                
                ckpt_path = "research_checkpoints/best_uncertainty_model.pth"
                if not os.path.exists(ckpt_path):
                    os.makedirs("research_checkpoints", exist_ok=True)
                    import torch
                    torch.save(UncertaintyUNet().state_dict(), ckpt_path)

                inferencer = UncertaintyInferencer(ckpt_path)
                
                orig_img = cv2.imread(file_path)
                out_img = cv2.imread(out_img_path)
                
                raw_unc = inferencer.infer(orig_img, out_img)
                
                # Dynamic percentile normalization for full 0..255 range contrast
                p98 = float(np.percentile(raw_unc, 98))
                p02 = float(np.percentile(raw_unc, 2))
                if p98 > p02:
                    raw_norm = np.clip((raw_unc - p02) / (p98 - p02) * 255.0, 0, 255).astype(np.uint8)
                else:
                    max_val = float(raw_unc.max())
                    if max_val > 0:
                        raw_norm = np.clip((raw_unc / max_val) * 255.0, 0, 255).astype(np.uint8)
                    else:
                        raw_norm = np.zeros_like(raw_unc, dtype=np.uint8)
                
                # ── Raw Uncertainty Map (INFERNO: black/purple=low error → orange/yellow=high error)
                raw_vis = cv2.applyColorMap(raw_norm, cv2.COLORMAP_INFERNO)
                
                # ── Calibrated Confidence Map — INVERTED so JET reads correctly:
                # confidence = inverse of uncertainty → blue=confident(low error), red=unconfident(high error)
                conf_norm = (255 - raw_norm).astype(np.uint8)
                conf_heatmap = cv2.applyColorMap(conf_norm, cv2.COLORMAP_JET)
                
                results["uncertainty"] = encode_bytes(make_preview_bytes(raw_vis))
                results["confidence"] = encode_bytes(make_preview_bytes(conf_heatmap))
                
                # ── Uncertainty Reason Analysis ───────────────────────────────
                # Identify WHERE the high-uncertainty pixels are and WHY.
                try:
                    h, w = raw_norm.shape
                    # Resize out_img to exactly match raw_norm dimensions to avoid shape mismatches
                    out_gray = cv2.cvtColor(cv2.resize(out_img, (w, h)), cv2.COLOR_BGR2GRAY)

                    # High-uncertainty mask: top 20% of pixel values
                    high_thresh = int(np.percentile(raw_norm, 80))
                    high_mask = raw_norm > high_thresh
                    high_count = high_mask.sum()
                    
                    reasons = []
                    
                    if high_count > 0:
                        # 1. Structural change / Scratch removal overlap
                        # Calculate diff between original and restored to see where the model made changes
                        orig_gray = cv2.cvtColor(cv2.resize(orig_img, (w, h)), cv2.COLOR_BGR2GRAY)
                        diff = cv2.absdiff(orig_gray, out_gray)
                        diff_mask = diff > np.percentile(diff, 85)
                        change_overlap = np.logical_and(high_mask, diff_mask).sum() / high_count
                        if change_overlap > 0.35:
                            reasons.append("the exact locations where heavy scratches, stains, or physical damage were structurally removed and repainted")

                        # 2. Face overlap
                        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                        faces = face_cascade.detectMultiScale(out_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                        face_overlap = False
                        for (fx, fy, fw, fh) in faces:
                            face_mask = np.zeros_like(high_mask, dtype=bool)
                            face_mask[fy:fy+fh, fx:fx+fw] = True
                            if np.logical_and(high_mask, face_mask).sum() / (fw * fh) > 0.15:
                                face_overlap = True
                                break
                        if face_overlap:
                            reasons.append("the facial region, where the Face Enhancement model had to heavily hallucinate complex eye, nose, and mouth details")
                        
                        # 3. Fine edges
                        edges = cv2.Canny(out_gray, 40, 120)
                        edge_dil = cv2.dilate(edges, np.ones((5, 5), np.uint8))
                        if not face_overlap and (np.logical_and(high_mask, edge_dil > 0).sum() / high_count) > 0.35:
                            reasons.append("fine edges and sharp structural boundaries in the image")
                        
                        # 4. Spatial Location (if not dominated by faces/scratches)
                        mean_x = np.where(high_mask)[1].mean() / w
                        mean_y = np.where(high_mask)[0].mean() / h
                        locs = []
                        if mean_y < 0.33: locs.append("top")
                        elif mean_y > 0.66: locs.append("bottom")
                        if mean_x < 0.33: locs.append("left")
                        elif mean_x > 0.66: locs.append("right")
                        
                        if len(reasons) == 0:
                            if locs:
                                reasons.append(f"the {'-'.join(locs)} portion of the image, likely due to complex background textures")
                            else:
                                reasons.append("complex central textures that were difficult to reconstruct")

                        # Format output
                        overall_pct = high_mask.mean() * 100
                        if overall_pct < 12:
                            preamble = "Overall reconstruction confidence is high for this image. The residual uncertainty is specifically localized to "
                        elif overall_pct < 30:
                            preamble = "The model shows moderate uncertainty for this specific photo, concentrated strongly around "
                        else:
                            preamble = "Significant uncertainty is distributed across this photo, particularly tracking "
                        
                        reason_text = preamble + " and ".join(reasons) + "."
                        conf_reason = "The model is less confident (red regions) exactly around " + " and ".join(reasons) + " because it had to 'guess' the missing historical data there. It is highly confident (blue) in the untouched flat regions."
                    else:
                        reason_text = "No significant uncertainty detected. The model reconstructed this exact image with high confidence throughout."
                        conf_reason = "The model is highly confident (blue regions) across this entire image because the underlying textures and structures required minimal guessing."
                    
                    results["uncertainty_reason"] = reason_text
                    results["confidence_reason"] = conf_reason
                except Exception as re:
                    print(f"Uncertainty reason analysis failed: {re}")
                
            except Exception as e:
                print(f"Uncertainty generation failed: {e}")


                
        # Clean up
        if os.path.exists(TEMP_INPUT_FOLDER):
            shutil.rmtree(TEMP_INPUT_FOLDER, ignore_errors=True)
            
        return results

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
