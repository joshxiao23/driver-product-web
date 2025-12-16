"""
Preprocessing utilities for Streamlit deployment.

Default behavior is designed to match your notebook's load_sequence_from_folder:
- resize only (no normalization) unless toggled
- optional ROI crop / CLAHE if you trained that way
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    import cv2
except Exception as e:
    cv2 = None


@dataclass
class PreprocessConfig:
    target_size: Tuple[int, int] = (224, 224)
    use_face_roi: bool = False
    use_clahe: bool = False
    normalize_0_1: bool = False


def _ensure_cv2():
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is not available. Ensure opencv-python-headless is installed.")


def _get_face_cascade():
    _ensure_cv2()
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            return None
        return face_cascade
    except Exception:
        return None


def apply_clahe_bgr(img_bgr: np.ndarray) -> np.ndarray:
    _ensure_cv2()
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)

    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    out = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    return out


def crop_face_roi(img_bgr: np.ndarray, target_size: Tuple[int, int], margin_ratio: float = 0.30) -> np.ndarray:
    _ensure_cv2()
    face_cascade = _get_face_cascade()
    if img_bgr is None:
        raise ValueError("Input image is None")

    if face_cascade is None:
        return cv2.resize(img_bgr, target_size)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return cv2.resize(img_bgr, target_size)

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    mh = int(h * margin_ratio)
    mw = int(w * margin_ratio)
    x1 = max(x - mw, 0)
    y1 = max(y - mh, 0)
    x2 = min(x + w + mw, img_bgr.shape[1])
    y2 = min(y + h + mh, img_bgr.shape[0])

    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return cv2.resize(img_bgr, target_size)

    return cv2.resize(roi, target_size)


def preprocess_frame_rgb(frame_rgb: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    Input: RGB image (H,W,3) uint8.
    Output: RGB float32 image resized to cfg.target_size.
    """
    _ensure_cv2()
    if frame_rgb is None:
        raise ValueError("frame_rgb is None")

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if cfg.use_face_roi:
        bgr = crop_face_roi(bgr, cfg.target_size)
    else:
        bgr = cv2.resize(bgr, cfg.target_size)

    if cfg.use_clahe:
        bgr = apply_clahe_bgr(bgr)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    if cfg.normalize_0_1:
        rgb = rgb / 255.0

    return rgb


def preprocess_sequence(frames_rgb: List[np.ndarray], cfg: PreprocessConfig, expected_len: int, pad_if_short: bool = True) -> np.ndarray:
    """
    Returns sequence array: (1, T, H, W, C)

    If fewer than expected_len frames are provided and pad_if_short=True,
    pads by repeating the last available frame to reach expected_len.
    """
    if len(frames_rgb) == 0:
        raise ValueError("No frames provided")

    frames = list(frames_rgb)

    if len(frames) < expected_len:
        if not pad_if_short:
            raise ValueError(f"Need at least {expected_len} frames, got {len(frames)}")
        last = frames[-1]
        while len(frames) < expected_len:
            frames.append(last)

    frames = frames[:expected_len]
    processed = [preprocess_frame_rgb(f, cfg) for f in frames]
    seq = np.stack(processed, axis=0)         # (T,H,W,C)
    seq = np.expand_dims(seq, axis=0)         # (1,T,H,W,C)
    return seq


def load_images_from_zip(uploaded_zip, max_frames: int) -> Tuple[List[np.ndarray], List[str]]:
    """
    Reads a ZIP from Streamlit uploader and returns (frames_rgb, filenames).
    Supports common image extensions.
    """
    _ensure_cv2()
    import zipfile
    import numpy as np

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    z = zipfile.ZipFile(uploaded_zip)
    names = [n for n in sorted(z.namelist()) if n.lower().endswith(valid_exts) and not n.endswith("/")]
    if not names:
        raise ValueError("No image files found in the ZIP.")

    names = names[:max_frames]
    frames = []
    out_names = []

    for name in names:
        data = z.read(name)
        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        out_names.append(name)

    if not frames:
        raise ValueError("Could not decode any images from the ZIP.")
    return frames, out_names
