import numpy as np
import cv2
import PySimpleGUI as sg
import os.path
import argparse
import os
import sys
import shutil
import traceback
from subprocess import call


def modify(image_filename=None, cv2_frame=None,
           with_scratch=False, with_colorize=False, with_hr=False,
           status_callback=None, saturation=1.0):
    """
    Run the full restoration (and optional colorization) pipeline.

    Parameters
    ----------
    image_filename  : path to a folder containing the input image(s)
    with_scratch    : enable scratch detection + removal
    with_colorize   : run Siggraph17 after global restoration
    with_hr         : enable high-resolution mode
    status_callback : callable(str) to push status messages (for GUI updates)
    saturation      : float multiplier for colorization saturation
    """

    def _status(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    def run_cmd(command):
        try:
            call(command, shell=True)
        except KeyboardInterrupt:
            print("Process interrupted")
            sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str,
                        default=image_filename, help="Test images")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./output",
        help="Restored images, please use the absolute path",
    )
    parser.add_argument("--GPU", type=str, default="-1", help="0,1,2")
    parser.add_argument(
        "--checkpoint_name", type=str, default="Setting_9_epoch_100",
        help="choose which checkpoint"
    )
    # Keep --with_scratch for CLI backward-compatibility but override with arg
    parser.add_argument("--with_scratch", action="store_true")
    opts = parser.parse_args([])   # parse empty — values come from GUI args above

    # Override opts with GUI values
    opts.input_folder = image_filename
    opts.output_folder = "./output"
    opts.GPU = "-1"
    opts.checkpoint_name = "Setting_9_epoch_100"
    opts.with_scratch = with_scratch

    gpu1 = opts.GPU

    # resolve relative paths before changing directory
    opts.input_folder = os.path.abspath(opts.input_folder)
    opts.output_folder = os.path.abspath(opts.output_folder)
    if not os.path.exists(opts.output_folder):
        os.makedirs(opts.output_folder)

    main_environment = os.path.dirname(os.path.abspath(__file__))
    py_cmd = f'"{sys.executable}"'

    try:
        # ── Stage 1: Overall Quality Improve ─────────────────────────────────
        _status("Stage 1/4: Restoring overall quality...")
        os.chdir(os.path.join(main_environment, "Global"))
        stage_1_input_dir = opts.input_folder
        stage_1_output_dir = os.path.join(
            opts.output_folder, "stage_1_restore_output")
        if os.path.exists(stage_1_output_dir):
            shutil.rmtree(stage_1_output_dir, ignore_errors=True)
        os.makedirs(stage_1_output_dir, exist_ok=True)

        if not opts.with_scratch:
            stage_1_command = (
                py_cmd + " test.py --test_mode Full --Quality_restore --test_input "
                + stage_1_input_dir
                + " --outputs_dir "
                + stage_1_output_dir
                + " --gpu_ids "
                + gpu1
            )
            run_cmd(stage_1_command)
        else:
            mask_dir = os.path.join(stage_1_output_dir, "masks")
            new_input = os.path.join(mask_dir, "input")
            new_mask = os.path.join(mask_dir, "mask")
            stage_1_command_1 = (
                py_cmd + " detection.py --test_path "
                + stage_1_input_dir
                + " --output_dir "
                + mask_dir
                + " --input_size full_size"
                + " --GPU "
                + gpu1
            )
            stage_1_command_2 = (
                py_cmd + " test.py --Scratch_and_Quality_restore --test_input "
                + new_input
                + " --test_mask "
                + new_mask
                + " --outputs_dir "
                + stage_1_output_dir
                + " --gpu_ids "
                + gpu1
            )
            run_cmd(stage_1_command_1)
            run_cmd(stage_1_command_2)

        # Fallback copy: if no faces are found later, stage_1 result goes to final_output
        stage_1_results = os.path.join(stage_1_output_dir, "restored_image")
        stage_4_output_dir = os.path.join(opts.output_folder, "final_output")
        if not os.path.exists(stage_4_output_dir):
            os.makedirs(stage_4_output_dir)
        for x in os.listdir(stage_1_results):
            shutil.copy(os.path.join(stage_1_results, x), stage_4_output_dir)

        print("Finish Stage 1 ...\n")

        # ── Stage 1b: Siggraph17 Colorization (optional) ─────────────────────
        if with_colorize:
            _status("Colorizing... (Siggraph17)")
            os.chdir(main_environment)   # back to project root for imports

            try:

                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                _status(f"Colorizing on {device}...")

                class SiggraphColorizer:
                    def __init__(self, device):
                        import sys
                        import os
                        colorizers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'colorization-master'))
                        if colorizers_path not in sys.path:
                            sys.path.append(colorizers_path)
                        from colorizers import siggraph17
                        self.device = device
                        self.model = siggraph17(pretrained=True).eval().to(device)
                        
                    def colorize_bgr(self, bgr_img, saturation=1.0):
                        import cv2
                        import numpy as np
                        import torch
                        from colorizers import preprocess_img, postprocess_tens
                        
                        # BGR to RGB
                        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                        
                        (tens_l_orig, tens_l_rs) = preprocess_img(rgb_img, HW=(256, 256))
                        tens_l_rs = tens_l_rs.to(self.device)
                        
                        with torch.no_grad():
                            out_ab = self.model(tens_l_rs).cpu()
                            
                            if saturation != 1.0:
                                out_ab = out_ab * saturation
                                
                        out_rgb_float = postprocess_tens(tens_l_orig, out_ab)
                        out_rgb_uint8 = (np.clip(out_rgb_float, 0.0, 1.0) * 255.0).astype(np.uint8)
                        
                        out_bgr = cv2.cvtColor(out_rgb_uint8, cv2.COLOR_RGB2BGR)
                        return out_bgr

                colorizer = SiggraphColorizer(device=device)

                colorize_output_dir = os.path.join(
                    opts.output_folder, "stage_1b_colorized")
                if os.path.exists(colorize_output_dir):
                    shutil.rmtree(colorize_output_dir, ignore_errors=True)
                os.makedirs(colorize_output_dir, exist_ok=True)

                colorized_files = []
                for fname in sorted(os.listdir(stage_1_results)):
                    src = os.path.join(stage_1_results, fname)
                    if not os.path.isfile(src):
                        continue
                    img = cv2.imread(src)
                    if img is None:
                        continue
                    try:
                        colorized = colorizer.colorize_bgr(img, saturation=saturation)
                    except MemoryError as me:
                        _status(f"Warning: {me}")
                        colorized = img   # fallback: use grayscale restored image
                    base = os.path.splitext(fname)[0]
                    dst = os.path.join(colorize_output_dir, f"{base}.png")
                    cv2.imwrite(dst, colorized)
                    colorized_files.append(dst)

                # Replace stage_1_results used by downstream stages with colorized
                # We DON'T modify stage_1_results path; instead we feed face stages
                # the colorized directory so faces are enhanced on colorized image.
                colorize_input_for_faces = colorize_output_dir

                # Also copy colorized images to final_output (overwrite gray copies)
                for f in colorized_files:
                    shutil.copy(f, stage_4_output_dir)

                _status("Colorization done.")

            except FileNotFoundError as fnf:
                _status(f"⚠ Colorization skipped: {fnf}")
                raise   # Re-raise so GUI can show proper error
            except Exception as exc:
                _status(f"⚠ Colorization error: {exc}")
                traceback.print_exc()
                raise
        else:
            colorize_input_for_faces = stage_1_results

        # ── Stage 2: Face Detection ───────────────────────────────────────────
        _status("Stage 2/4: Detecting faces...")
        os.chdir(os.path.join(main_environment, "Face_Detection"))
        stage_2_input_dir = colorize_input_for_faces
        stage_2_output_dir = os.path.join(
            opts.output_folder, "stage_2_detection_output")
        if os.path.exists(stage_2_output_dir):
            shutil.rmtree(stage_2_output_dir, ignore_errors=True)
        os.makedirs(stage_2_output_dir, exist_ok=True)

        detect_script = "detect_all_dlib_HR.py" if with_hr else "detect_all_dlib.py"
        stage_2_command = (
            py_cmd + f" {detect_script} --url " + stage_2_input_dir
            + " --save_url " + stage_2_output_dir
        )
        run_cmd(stage_2_command)
        print("Finish Stage 2 ...\n")

        # ── Stage 3: Face Enhancement ─────────────────────────────────────────
        _status("Stage 3/4: Enhancing faces...")
        os.chdir(os.path.join(main_environment, "Face_Enhancement"))
        stage_3_input_mask = "./"
        stage_3_input_face = stage_2_output_dir
        stage_3_output_dir = os.path.join(
            opts.output_folder, "stage_3_face_output")
        if os.path.exists(stage_3_output_dir):
            shutil.rmtree(stage_3_output_dir, ignore_errors=True)
        os.makedirs(stage_3_output_dir, exist_ok=True)
        if with_hr:
            ckpt_name = 'FaceSR_512'
            load_size = '512'
            batch_size = '1'
        else:
            ckpt_name = opts.checkpoint_name
            load_size = '256'
            batch_size = '4'

        stage_3_command = (
            py_cmd + " test_face.py --old_face_folder "
            + stage_3_input_face
            + " --old_face_label_folder "
            + stage_3_input_mask
            + " --tensorboard_log --name "
            + ckpt_name
            + " --gpu_ids "
            + gpu1
            + f" --load_size {load_size} --label_nc 18 --no_instance --preprocess_mode resize --batchSize {batch_size} --results_dir "
            + stage_3_output_dir
            + " --no_parsing_map"
        )
        run_cmd(stage_3_command)
        print("Finish Stage 3 ...\n")

        # ── Stage 4: Warp back + Blending ─────────────────────────────────────
        _status("Stage 4/4: Blending and finalizing...")
        os.chdir(os.path.join(main_environment, "Face_Detection"))
        stage_4_input_image_dir = colorize_input_for_faces
        stage_4_input_face_dir = os.path.join(stage_3_output_dir, "each_img")
        stage_4_output_dir = os.path.join(opts.output_folder, "final_output")
        if not os.path.exists(stage_4_output_dir):
            os.makedirs(stage_4_output_dir)

        warp_script = "align_warp_back_multiple_dlib_HR.py" if with_hr \
                      else "align_warp_back_multiple_dlib.py"
        stage_4_command = (
            py_cmd + f" {warp_script} --origin_url "
            + stage_4_input_image_dir
            + " --replace_url "
            + stage_4_input_face_dir
            + " --save_url "
            + stage_4_output_dir
        )
        run_cmd(stage_4_command)
        print("Finish Stage 4 ...\n")

        _status("All processing done!")

    finally:
        os.chdir(main_environment)


# ── GUI ───────────────────────────────────────────────────────────────────────

def make_preview_bytes(img, max_dim=450):
    if img is None:
        return b""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return cv2.imencode('.png', img)[1].tobytes()


def main():
    # ── Layout ─────────────────────────────────────────────────────────────────
    
    images_col = [
        # File picker
        [
            sg.Text('Input file:'),
            sg.In(enable_events=True, key='-IN FILE-'),
            sg.FileBrowse(file_types=(("Image Files", "*.png;*.jpg;*.jpeg;*.bmp"),)),
        ],
        # Options
        [sg.HorizontalSeparator()],
        [sg.Text('Options:', font=('Helvetica', 10, 'bold'))],
        [
            sg.Checkbox('Scratch Restoration', key='-SCRATCH-', default=False),
            sg.Checkbox('Colorize (Siggraph17)', key='-COLORIZE-', default=False),
            sg.Checkbox('High Resolution (HR)', key='-HR-', default=False),
        ],
        [
            sg.Checkbox('Reliability Analysis (Uncertainty)', key='-UNCERTAINTY-', default=False)
        ],
        [
            sg.Text('Color Saturation:'),
            sg.Slider(range=(0.0, 2.0), default_value=1.0, resolution=0.1, orientation='h', size=(20, 15), key='-SATURATION-')
        ],
        [sg.HorizontalSeparator()],
        # Buttons
        [
            sg.Button('Restore Photo', key='-MPHOTO-'),
            sg.Button('Open Output Folder', key='-OPEN_OUT-'),
            sg.Button('Exit'),
        ],
        # Status bar
        [sg.Text('', key='-STATUS-', size=(80, 2), text_color='lightgreen')],
        # Image preview area
        [
            sg.Column([[sg.Text("Original")], [sg.Image(filename='', key='-IN-')]], element_justification='c'),
            sg.Column([[sg.Text("Restored")], [sg.Image(filename='', key='-OUT-')]], element_justification='c'),
            sg.Column([[sg.Text("Raw Uncertainty")], [sg.Image(filename='', key='-UNC-')]], element_justification='c'),
            sg.Column([[sg.Text("Calibrated Confidence")], [sg.Image(filename='', key='-CONF-')]], element_justification='c')
        ]
    ]
    
    layout = [[sg.Column(images_col, element_justification='c', expand_x=True, expand_y=True)]]
    
    window = sg.Window('Bringing Old Photos Back to Life', layout, grab_anywhere=True, resizable=True)
    
    prev_filename = None
    filename = None
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    
    # ── Event loop ────────────────────────────────────────────────────────────────
    
    while True:
        event, values = window.read()
    
        if event in (None, 'Exit'):
            break
    
        elif event == '-OPEN_OUT-':
            out_dir = os.path.join(project_root, "output", "final_output")
            os.makedirs(out_dir, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(out_dir)
            else:
                call(["xdg-open", out_dir])
    

    
        elif event == '-MPHOTO-':
            if not filename or not os.path.isfile(filename):
                window['-STATUS-'].update("Please select a valid image file first.")
                continue
    
            # Read GUI option checkboxes
            do_scratch  = values['-SCRATCH-']
            do_colorize = values['-COLORIZE-']
            do_hr       = values['-HR-']
            do_uncertainty = values['-UNCERTAINTY-']
    
            try:
                # Build human-readable status description
                stages = ["Restoring"]
                if do_scratch:
                    stages.append("Scratch removal")
                if do_colorize:
                    stages.append("Colorizing")
                stages.append("Face enhancement")
                window['-STATUS-'].update("Running: " + " → ".join(stages) + "...")
                window.refresh()
    
                # Ensure project root on sys.path so Colorization can be imported
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
    
                # Set up a clean temporary folder for this single input image
                temp_input_folder = os.path.join(project_root, "gui_temp_input")
                if os.path.exists(temp_input_folder):
                    shutil.rmtree(temp_input_folder)
                os.makedirs(temp_input_folder, exist_ok=True)
    
                base_name = os.path.basename(filename)
                shutil.copy(filename, os.path.join(temp_input_folder, base_name))
    
                output_folder = os.path.join(project_root, "output")
    
                def _gui_status(msg):
                    window['-STATUS-'].update(msg)
                    window.refresh()
    
                modify(
                    image_filename=temp_input_folder,
                    with_scratch=do_scratch,
                    with_colorize=do_colorize,
                    with_hr=do_hr,
                    status_callback=_gui_status,
                    saturation=values['-SATURATION-'],
                )
    
                # Find and display the output image
                base_name_no_ext = os.path.splitext(base_name)[0]
                final_dir = os.path.join(output_folder, "final_output")
    
                candidates = [
                    os.path.join(final_dir, f"{base_name_no_ext}.png"),
                    os.path.join(final_dir, base_name),
                ]
                f_image = None
                for cand in candidates:
                    if os.path.exists(cand):
                        f_image = cand
                        break
    
                if f_image and os.path.exists(f_image):
                    out_img = cv2.imread(f_image)
                    if out_img is not None:
                        window['-OUT-'].update(data=make_preview_bytes(out_img))
                        
                        if do_uncertainty:
                            try:
                                _gui_status("Running Uncertainty Analysis...")
                                from research.uncertainty import UncertaintyInferencer
                                from research.calibration import UncertaintyCalibrator
                                
                                ckpt_path = os.path.join(project_root, "research_checkpoints", "best_uncertainty_model.pth")
                                calib_path = os.path.join(project_root, "research_checkpoints", "calibrator.pkl")
                                
                                inferencer = UncertaintyInferencer(ckpt_path)
                                calibrator = UncertaintyCalibrator.load(calib_path)
                                
                                orig_img = cv2.imread(filename)
                                raw_unc = inferencer.infer(orig_img, out_img)
                                calib_unc = calibrator.calibrate(raw_unc)
                                
                                p98 = float(np.percentile(calib_unc, 98))
                                p02 = float(np.percentile(calib_unc, 2))
                                if p98 > p02:
                                    norm = np.clip((calib_unc - p02) / (p98 - p02) * 255.0, 0, 255).astype(np.uint8)
                                else:
                                    max_val = float(calib_unc.max())
                                    norm = np.clip((calib_unc / max(1e-5, max_val)) * 255.0, 0, 255).astype(np.uint8)
                                
                                raw_vis = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
                                conf_heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                                
                                window['-UNC-'].update(data=make_preview_bytes(raw_vis))
                                window['-CONF-'].update(data=make_preview_bytes(conf_heatmap))
                            except Exception as e:
                                _gui_status(f"Uncertainty Analysis failed: {e}")
                        else:
                            # Clear if disabled
                            window['-UNC-'].update(data=b"")
                            window['-CONF-'].update(data=b"")
                            
                        label = "restored + colorized" if do_colorize else "restored"
                        window['-STATUS-'].update(
                            f"Done! {label.capitalize()}: {os.path.basename(f_image)}")
                    else:
                        window['-STATUS-'].update(
                            "Finished, but could not decode output image.")
                else:
                    window['-STATUS-'].update(
                        "Finished. Please check the output folder.")
    
                if os.path.exists(temp_input_folder):
                    shutil.rmtree(temp_input_folder, ignore_errors=True)
    
            except FileNotFoundError as fnf:
                # Clean error (e.g. missing checkpoint)
                window['-STATUS-'].update(f"Error: {fnf}")
            except Exception as e:
                window['-STATUS-'].update(f"Error: {e}")
                traceback.print_exc()
    
        elif event == '-IN FILE-':
            filename = values['-IN FILE-']
            if filename != prev_filename:
                prev_filename = filename
                try:
                    if filename and os.path.isfile(filename):
                        image = cv2.imread(filename)
                        if image is not None:
                            window['-IN-'].update(data=make_preview_bytes(image))
                            window['-STATUS-'].update(
                                f"Loaded: {os.path.basename(filename)}")
                except Exception as e:
                    window['-STATUS-'].update(f"Could not load image: {e}")
    
    window.close()

if __name__ == '__main__':
    main()
