from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

import tensorflow as tf


def saliency_heatmap_for_sequence(model, seq_array: np.ndarray, frame_index: int = 0) -> np.ndarray:
    if seq_array is None:
        raise ValueError("seq_array is None")

    x = tf.convert_to_tensor(seq_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x, training=False)
        score = y[:, 0]  # for binary sigmoid

    grads = tape.gradient(score, x)  # (1,T,H,W,C)
    if grads is None:
        raise RuntimeError("Could not compute gradients for saliency map.")

    grads = tf.abs(grads)[0, frame_index]          # (H,W,C)
    heat = tf.reduce_mean(grads, axis=-1).numpy()  # (H,W)

    heat = np.maximum(heat, 0)
    mx = float(np.max(heat))
    if mx > 0:
        heat = heat / mx
    return heat.astype(np.float32)


def overlay_heatmap_on_frame(frame_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if cv2 is None:
        raise ImportError("cv2 not available for heatmap overlay.")
    if frame_rgb is None:
        raise ValueError("frame_rgb is None")

    h, w = frame_rgb.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm_uint8 = np.uint8(255 * hm)

    colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    overlay = (alpha * colored + (1 - alpha) * frame_rgb).astype(np.uint8)
    return overlay
