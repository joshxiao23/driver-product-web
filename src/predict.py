from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

import numpy as np
from tensorflow import keras


@lru_cache(maxsize=2)
def load_model_cached(model_path: str):
    if not model_path or not isinstance(model_path, str):
        raise ValueError("model_path must be a non-empty string")
    model = keras.models.load_model(model_path, compile=False, safe_mode=False)
    return model


def predict_sequence_array(model, seq_array: np.ndarray, class_names: List[str], threshold: float = 0.5) -> Dict:
    if seq_array is None:
        raise ValueError("seq_array is None")

    x = np.asarray(seq_array, dtype=np.float32)
    y = np.asarray(model.predict(x, verbose=0))
    raw = {"shape": list(y.shape), "values": y.flatten()[:10].tolist()}

    # Sigmoid binary: (1,1)
    if y.ndim == 2 and y.shape[1] == 1:
        p_pos = float(y[0, 0])
        p_pos = max(0.0, min(1.0, p_pos))
        p_neg = 1.0 - p_pos

        # If 2 names provided, map [0]=negative, [1]=positive
        if len(class_names) >= 2:
            neg_name = class_names[0]
            pos_name = class_names[1]
        else:
            neg_name, pos_name = "class_0", "class_1"

        label = pos_name if p_pos >= threshold else neg_name
        return {"label": label, "probs": {neg_name: round(p_neg, 6), pos_name: round(p_pos, 6)}, "raw": raw, "threshold": float(threshold)}

    # Softmax/multi-class: (1,N)
    if y.ndim == 2 and y.shape[1] >= 2:
        probs = y[0].astype(np.float64)
        s = probs.sum()
        if s > 0:
            probs = probs / s

        n = probs.shape[0]
        names = class_names[:n] if len(class_names) >= n else [f"class_{i}" for i in range(n)]
        idx = int(np.argmax(probs))
        label = names[idx]
        return {"label": label, "probs": {names[i]: round(float(probs[i]), 6) for i in range(n)}, "raw": raw, "threshold": float(threshold)}

    raise ValueError(f"Unsupported model output shape: {y.shape}. Raw={raw}")
