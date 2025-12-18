import io
import os
from datetime import datetime

import numpy as np
import streamlit as st

import tensorflow as tf
st.sidebar.write("TensorFlow version:", tf.__version__)

from src.preprocess import (
    PreprocessConfig,
    load_images_from_zip,
    preprocess_sequence,
)
from src.predict import (
    load_model_cached,
    predict_sequence_array,
)
from src.gradcam import (
    saliency_heatmap_for_sequence,
    overlay_heatmap_on_frame,
)
from src.reports import (
    build_report_markdown,
    save_text_report,
)

st.set_page_config(
    page_title="Driver Behavior Detection (CNN+RNN)",
    page_icon="🚗",
    layout="wide",
)

APP_TITLE = "Driver Behavior Detection (CNN+RNN)"
DEFAULT_CLASS_NAMES = ["drowsy", "notdrowsy"]  # edit if yours are different

st.title(APP_TITLE)
st.caption("Upload a ZIP of frames (images). The app runs CNN+RNN inference, shows probabilities, a saliency explanation, and a report placeholder.")

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Model & Settings")

model_path = st.sidebar.text_input(
    "Model path (.keras/.h5) inside repo",
    value="models/seq_model.h5",
    help="Put your trained CNN+RNN model file inside /models. Example: models/seq_model.h5",
)

st.sidebar.subheader("Debug (model file)")
try:
    if os.path.exists(model_path):
        st.sidebar.write("Exists:", True)
        st.sidebar.write("Size (bytes):", os.path.getsize(model_path))
        with open(model_path, "rb") as f:
            head = f.read(120)
        st.sidebar.code(head.decode("utf-8", errors="replace"))
    else:
        st.sidebar.write("Exists:", False)
except Exception as e:
    st.sidebar.error(str(e))

class_names = st.sidebar.text_input(
    "Class names (comma-separated)",
    value=",".join(DEFAULT_CLASS_NAMES),
    help="Match the order used during training. Example: drowsy,notdrowsy",
    
)
class_names = [c.strip() for c in class_names.split(",") if c.strip()]

threshold = st.sidebar.slider(
    "Decision threshold (for positive class)",
    min_value=0.0, max_value=1.0, value=0.5, step=0.05
)

st.sidebar.divider()
st.sidebar.header("Preprocessing")

cfg = PreprocessConfig(
    # IMPORTANT: default OFF to match your notebook's load_sequence_from_folder (no ROI crop).
    use_face_roi=st.sidebar.checkbox(
        "Crop face ROI",
        value=False,
        help="Turn ON only if your model was trained on face-cropped frames."
    ),
    use_clahe=st.sidebar.checkbox("contrast enhancement", value=False),
    normalize_0_1=st.sidebar.checkbox(
        "Normalize",
        value=False,
        help="Turn ON only if your model was trained with 0–1 scaled pixels."
    ),
    target_size=(224, 224),
)

sequence_length = st.sidebar.number_input(
    "Sequence length",
    min_value=4, max_value=64, value=8, step=1,
    help="Must match the sequence length expected by your trained model."
)

st.sidebar.divider()
st.sidebar.header("Debug")
if st.sidebar.checkbox("Show model input/output shapes", value=False):
    try:
        m = load_model_cached(model_path)
        st.sidebar.write("Model input shape:", getattr(m, "input_shape", None))
        st.sidebar.write("Model output shape:", getattr(m, "output_shape", None))
    except Exception as e:
        st.sidebar.error(f"Model load/debug failed: {e}")

st.sidebar.divider()
st.sidebar.header("Help")
with st.sidebar.expander("How to use this app"):
    st.markdown(
        """
**1) Prepare your input**  
- Extract frames from a short video clip (e.g., 0001.jpg ... 0016.jpg).  
- Zip the frames (ZIP should contain only images).

**2) Upload**  
- Upload the ZIP in the 'Upload & Predict' tab.

**3) Predict**  
- The app will preprocess, run CNN+RNN prediction, and show probabilities.

**4) Explanation**  
- Generates a gradient-based saliency overlay for a selected frame.

**5) Reports (placeholder)**  
- Generates a simple placeholder report and allows download.
        """
    )

tabs = st.tabs([
    "1) Upload & Predict",
    "2) Explanation (Saliency)",
    "3) Reports (Placeholder)",
    "4) Multi-user note",
])

# ---------------------------
# Session state
# ---------------------------
for k in ["seq_array", "orig_frames", "file_names", "pred", "heatmap", "overlay"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ===========================
# TAB 1
# ===========================
with tabs[0]:
    st.subheader("Upload a ZIP of frames and run prediction")

    uploaded = st.file_uploader("Upload ZIP (frames only)", type=["zip"], accept_multiple_files=False)

    colA, colB = st.columns([1, 1])

    with colA:
        st.write("**Preview & preprocess**")
        if uploaded is not None:
            try:
                frames, names = load_images_from_zip(uploaded, max_frames=int(sequence_length))
                st.session_state.orig_frames = frames
                st.session_state.file_names = names

                st.success(f"Loaded {len(frames)} frames from ZIP.")
                st.caption("Preview (first frame):")
                st.image(frames[0], caption=names[0], use_container_width=True)

                # pad_if_short=True prevents shape mismatch if the ZIP has fewer frames
                seq = preprocess_sequence(frames, cfg, expected_len=int(sequence_length), pad_if_short=True)
                st.session_state.seq_array = seq
                st.info(f"Preprocessed sequence shape: {seq.shape} (batch, time, H, W, C)")

            except Exception as e:
                st.error(f"Could not read ZIP / preprocess: {e}")

    with colB:
        st.write("**Run inference (CNN+RNN)**")
        run_btn = st.button("Run sequence prediction", type="primary", use_container_width=True)

        if run_btn:
            if st.session_state.seq_array is None:
                st.warning("Upload a ZIP first (left side).")
            else:
                try:
                    model = load_model_cached(model_path)

                    pred = predict_sequence_array(
                        model=model,
                        seq_array=st.session_state.seq_array,
                        class_names=class_names,
                        threshold=float(threshold),
                    )
                    st.session_state.pred = pred

                    st.success("Prediction complete.")
                    st.markdown(f"**Predicted label:** `{pred['label']}`")
                    st.write("**Probabilities:**")
                    st.json(pred["probs"])

                    with st.expander("Raw model output (debug)"):
                        st.json(pred.get("raw", {}))

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.info(
                        "Common causes: (1) wrong model path, (2) sequence length T doesn't match the model, "
                        "(3) your model outputs softmax but class_names/threshold mapping differs, "
                        "(4) training preprocessing differs (normalization/ROI/size)."
                    )

# ===========================
# TAB 2
# ===========================
with tabs[1]:
    st.subheader("Gradient-based saliency explanation")

    if st.session_state.pred is None or st.session_state.seq_array is None or st.session_state.orig_frames is None:
        st.info("Run a prediction first in Tab 1.")
    else:
        frame_idx = st.slider(
            "Frame index to explain",
            min_value=0,
            max_value=max(0, int(sequence_length) - 1),
            value=0,
            step=1
        )

        explain_btn = st.button("Generate saliency overlay", use_container_width=True)

        if explain_btn:
            try:
                model = load_model_cached(model_path)
                heatmap = saliency_heatmap_for_sequence(model=model, seq_array=st.session_state.seq_array, frame_index=int(frame_idx))
                st.session_state.heatmap = heatmap

                frame = st.session_state.orig_frames[min(int(frame_idx), len(st.session_state.orig_frames)-1)]
                overlay = overlay_heatmap_on_frame(frame, heatmap)
                st.session_state.overlay = overlay

                st.success("Explanation generated.")
            except Exception as e:
                st.error(f"Explanation failed: {e}")

        if st.session_state.overlay is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.write("Original frame")
                st.image(st.session_state.orig_frames[min(int(frame_idx), len(st.session_state.orig_frames)-1)], use_container_width=True)
            with c2:
                st.write("Saliency overlay")
                st.image(st.session_state.overlay, use_container_width=True)

            # Download overlay
            import PIL.Image
            img = PIL.Image.fromarray(st.session_state.overlay)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.download_button(
                "Download explanation PNG",
                data=buf.getvalue(),
                file_name=f"saliency_overlay_frame{frame_idx}.png",
                mime="image/png",
                use_container_width=True
            )

# ===========================
# TAB 3
# ===========================
with tabs[2]:
    st.subheader("Report generation placeholder")

    st.markdown(
        """
This section is a **placeholder** for web-based report generation.
It shows **where** reports will appear, and demonstrates a report being requested, produced, and presented.
        """
    )

    if st.session_state.pred is None:
        st.info("Run a prediction first in Tab 1.")
    else:
        if st.button("Generate report (placeholder)", use_container_width=True):
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            md = build_report_markdown(
                timestamp=now,
                model_path=model_path,
                pred=st.session_state.pred,
                class_names=class_names,
                sequence_length=int(sequence_length),
            )
            st.markdown(md)

            os.makedirs("outputs", exist_ok=True)
            out_path = os.path.join("outputs", f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
            save_text_report(out_path, md)

            st.download_button(
                "Download report (.md)",
                data=md.encode("utf-8"),
                file_name=os.path.basename(out_path),
                mime="text/markdown",
                use_container_width=True
            )

# ===========================
# TAB 4
# ===========================
with tabs[3]:
    st.subheader("Multiple simultaneous users")
    st.markdown(
        """
Streamlit apps are web-hosted, so multiple users can access the same URL at the same time.

This app uses `st.session_state`, so each user's uploads and results are isolated per session.
        """
    )
