# webcam_pil_demo.py
import argparse
import time
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from app import FaceDetLiteApp
from qai_hub_models.models.face_det_lite.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    FaceDetLite,
)

def parse_args():
    p = argparse.ArgumentParser(description="Show USB webcam with a PIL processing hook + overlay.")
    p.add_argument("--device", default="0",
                   help="Camera index (e.g., 0) or V4L2 path (e.g., /dev/video3).")
    p.add_argument("--width", type=int, default=0, help="Requested capture width (e.g., 1280).")
    p.add_argument("--height", type=int, default=0, help="Requested capture height (e.g., 720).")
    p.add_argument("--mirror", action="store_true", help="Mirror preview horizontally.")
    return p.parse_args()

def open_capture(device_str: str):
    """
    Tries to open either a numeric index or a V4L2 device path.
    """
    cap = None
    # 1) Numeric index (macOS/Windows/Linux)
    try:
        idx = int(device_str)
        cap = cv2.VideoCapture(idx)  # CAP_ANY: AVFoundation on macOS, MSMF/DSHOW on Windows, V4L2 on Linux
    except ValueError:
        # Not an int -> maybe a V4L2 path like /dev/video3
        # Ask OpenCV to use V4L2 backend (safe no-op on non-Linux builds).
        cap = cv2.VideoCapture(device_str, cv2.CAP_V4L2)

    if not cap or not cap.isOpened():
        # Last-ditch: if it *looks* like /dev/videoN, try N as index.
        if device_str.startswith("/dev/video"):
            tail_digits = "".join([c for c in device_str if c.isdigit()])
            if tail_digits.isnumeric():
                alt_idx = int(tail_digits)
                cap = cv2.VideoCapture(alt_idx)
    if not cap or not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera '{device_str}'. "
            "Tips: On Linux, pass a valid /dev/videoN; on macOS/Windows, pass a numeric index like 0."
        )
    return cap

def request_size(cap, w, h):
    if w > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    if h > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

def draw_overlay_with_pil(pil_img: Image.Image, fps: float) -> Image.Image:
    """
    Example overlay using Pillow:
      - translucent banner with timestamp + FPS
      - crosshair
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    text = f"{ts}   FPS: {fps:.1f}"

    # Work in RGBA for alpha blending
    base = pil_img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    W, H = base.size

    # Text box
    pad = 8
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    box_w, box_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([0, 0, box_w + 2 * pad, box_h + 2 * pad], fill=(0, 0, 0, 128))
    draw.text((pad, pad), text, fill=(255, 255, 255, 255), font=font)

    # Crosshair in the center
    cx, cy = W // 2, H // 2
    line_len = min(W, H) // 10
    draw.line([(cx - line_len, cy), (cx + line_len, cy)], width=2, fill=(0, 255, 0, 255))
    draw.line([(cx, cy - line_len), (cx, cy + line_len)], width=2, fill=(0, 255, 0, 255))

    # Composite overlay and return to RGB
    out = Image.alpha_composite(base, overlay).convert("RGB")
    return out

# --- Your hook: put your own PIL processing here ------------------------------
def process_frame_pil(pil_img: Image.Image) -> Image.Image:
    """
    This function is called for every frame as a PIL.Image in RGB mode.
    Modify it as you like and return the result.
    For demo purposes we just call the overlay helper.
    """
    # Example: you could run filters, paste sprites, run OCR, etc.
    # Return the modified PIL image.
    return pil_img
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    cap = open_capture(args.device)
    request_size(cap, args.width, args.height)

    win_name = "USB Cam (PIL pipeline demo) - press 'q' to quit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # FPS measurement
    last = time.time()
    fps = 0.0
    smoothing = 0.9  # exponential moving average

    model = demo_model_from_cli_args(FaceDetLite, MODEL_ID, args)

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Failed to read frame from camera.", file=sys.stderr)
                break

            if args.mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)

            # ---- OpenCV BGR -> PIL RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # ---- Your per-frame PIL processing hook
            app = FaceDetLiteApp(model)
            res, pil_img = app.run_inference_on_image(pil_img, 0)
            # print(f"Image resolution: {pil_img.size}")
            pil_img = process_frame_pil(pil_img)

            # ---- Demo overlay (kept separate so you can remove it if you like)
            # Compute instantaneous FPS before drawing
            # now = time.time()
            # inst_fps = 1.0 / max(1e-6, (now - last))
            # fps = smoothing * fps + (1 - smoothing) * inst_fps
            # last = now
            # pil_img = draw_overlay_with_pil(pil_img, fps=fps)

            # ---- PIL RGB -> OpenCV BGR for display
            frame_rgb_out = np.array(pil_img)              # still RGB
            frame_bgr_out = cv2.cvtColor(frame_rgb_out, cv2.COLOR_RGB2BGR)

            cv2.imshow(win_name, frame_bgr_out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
