"""
Embedded Night-Vision System for Pedestrian Detection
=====================================================
A premium Streamlit web UI for real-time pedestrian detection
on night-vision / low-light images using YOLOv2 and Haar+AdaBoost.

Author : <Your Name>
Stack  : Python 3.11 · OpenCV 4.x · Streamlit 1.x · NumPy
"""
import download_weights
import sys
import time
from pathlib import Path

import cv2
import imutils
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Project root – so we can locate model files regardless of CWD
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "yolov2model"

# ---------------------------------------------------------------------------
# Page config (MUST be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Night-Vision Pedestrian Detection",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS – dark glassmorphism theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ---------- global ---------- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
    }

    /* ---------- sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.85);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* ---------- glass card ---------- */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(16px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    }

    /* ---------- hero ---------- */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: rgba(255,255,255,0.55);
        margin-bottom: 24px;
        line-height: 1.6;
    }

    /* ---------- metric cards ---------- */
    .metric-row {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        margin-bottom: 20px;
    }
    .metric-card {
        flex: 1;
        min-width: 140px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #a78bfa;
    }
    .metric-card .label {
        font-size: 0.82rem;
        color: rgba(255,255,255,0.45);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* ---------- badges ---------- */
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-purple { background: rgba(167,139,250,0.18); color: #a78bfa; border: 1px solid rgba(167,139,250,0.25); }
    .badge-blue   { background: rgba(96,165,250,0.18);  color: #60a5fa; border: 1px solid rgba(96,165,250,0.25);  }
    .badge-green  { background: rgba(52,211,153,0.18);  color: #34d399; border: 1px solid rgba(52,211,153,0.25);  }
    .badge-amber  { background: rgba(251,191,36,0.18);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.25);  }

    /* ---------- footer ---------- */
    .footer {
        text-align: center;
        padding: 32px 0 16px 0;
        color: rgba(255,255,255,0.25);
        font-size: 0.82rem;
    }

    /* ---------- tweak file uploader ---------- */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(167,139,250,0.3) !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }

    /* ---------- buttons ---------- */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(124, 58, 237, 0.45);
    }

    /* ---------- download button ---------- */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #059669 0%, #34d399 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 28px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(5, 150, 105, 0.3);
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(5, 150, 105, 0.45);
    }

    /* ---------- radio buttons ---------- */
    .stRadio > div { gap: 8px; }

    /* hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
#  DETECTION BACK-END
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_yolo_model():
    """Load YOLOv2 Darknet model and class labels (cached across sessions)."""
    cfg_path = str(MODEL_DIR / "yolov2.cfg")
    weights_path = str(MODEL_DIR / "yolov2.weights")
    labels_path = str(MODEL_DIR / "yolov2-labels")

    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    class_labels = open(labels_path).read().strip().split("\n")
    return net, layer_names, class_labels


def _list_bounding_boxes(outputs, h, w, conf_thresh=0.5):
    """Parse YOLO outputs into bounding boxes, confidences, and class IDs."""
    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > conf_thresh:
                box = detection[0:4] * np.array([w, h, w, h])
                cx, cy, bw, bh = box.astype("int")
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)
                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(confidence)
                class_ids.append(class_id)
    return boxes, confidences, class_ids


def detect_yolo(image: np.ndarray, conf_thresh: float = 0.5, nms_thresh: float = 0.3, gamma: float = 1.0):
    """
    Run YOLOv2 pedestrian detection on *image* (BGR numpy array).

    Returns
    -------
    annotated : np.ndarray   – image with bounding boxes drawn
    count     : int           – number of detections
    details   : list[dict]    – per-detection info (label, confidence, box)
    """
    net, layer_names, class_labels = load_yolo_model()
    h, w = image.shape[:2]
    label_colors = np.random.RandomState(42).randint(0, 255, size=(len(class_labels), 3), dtype="uint8")

    # Apply optional gamma correction before creating blob
    if gamma != 1.0:
        image = _adjust_gamma(image, gamma=gamma)
    # Use larger input size for better detection accuracy
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (608, 608), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    boxes, confidences, class_ids = _list_bounding_boxes(outputs, h, w, conf_thresh)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)

    annotated = image.copy()
    details = []

    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, bw, bh = boxes[i]
            color = [int(c) for c in label_colors[class_ids[i]]]
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
            label_text = f"{class_labels[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(annotated, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            details.append({
                "label": class_labels[class_ids[i]],
                "confidence": round(confidences[i], 4),
                "box": [x, y, bw, bh],
            })

    return annotated, len(details), details


def _adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply gamma correction to brighten / darken an image."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def detect_haar_adaboost(image: np.ndarray, gamma: float = 3.5):
    """
    Run Haar + AdaBoost pedestrian detection pipeline optimized for low‑light images.

    Pipeline:
    1. Gamma correction to brighten the image.
    2. Convert to grayscale (required for HOG detector).
    3. Optional contrast enhancement via CLAHE.
    4. Detect pedestrians using HOG + SVM (default people detector).

    Returns
    -------
    annotated : np.ndarray – image with detections drawn
    count     : int       – number of detections
    details   : list[dict] – per‑detection info (label, confidence, box)
    """
    # 1. Gamma correction for low‑light enhancement
    adjusted = _adjust_gamma(image, gamma=gamma)
    # 2. Convert to grayscale (HOG expects single channel)
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    # 3. Apply CLAHE for contrast improvement (helps under‑exposed regions)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    gray_clahe = clahe.apply(gray)
    # 4. Initialize HOG descriptor with the default people detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Detect multi‑scale pedestrians
    regions, _ = hog.detectMultiScale(
        gray_clahe,
        winStride=(4, 4),
        padding=(8, 8),
        scale=1.05,
    )

    # Prepare annotated image (use the gamma‑adjusted colour image for visual output)
    annotated = adjusted.copy()
    details = []
    for (x, y, w, h) in regions:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        details.append({
            "label": "person",
            "confidence": None,
            "box": [int(x), int(y), int(w), int(h)],
        })

    return annotated, len(details), details
            "confidence": None,
            "box": [int(x), int(y), int(w), int(h)],
        })

    return annotated, len(details), details


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; margin-bottom:20px;">
            <span style="font-size:2.4rem;">🌙</span>
            <h2 style="margin:4px 0 0 0; font-size:1.15rem; font-weight:600;
                background: linear-gradient(135deg,#a78bfa,#60a5fa);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                Night-Vision Detection
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("#### ⚙️ Detection Settings")

    method = st.selectbox(
        "Algorithm",
        ["YOLOv2 (Deep Learning)", "Haar + AdaBoost (Classical CV)"],
        help="YOLOv2 is more accurate; Haar+AdaBoost is faster on CPU.",
    )

    conf_thresh = st.slider(
    "Confidence threshold",
    0.1, 1.0, 0.3, 0.05,
    help="Lower threshold helps detect dim pedestrians (YOLOv2).",
    disabled="Haar" in method,
)    )

    nms_thresh = st.slider(
    "NMS threshold",
    0.1, 1.0, 0.4, 0.05,
    help="Increase to keep overlapping boxes when needed (YOLOv2 only).",
    disabled="Haar" in method,
)

    # Gamma correction slider for Haar (default) and optional for YOLO
    gamma_val = st.slider(
        "Gamma correction (Haar)",
        0.5, 5.0, 2.2, 0.1,
        help="Adjusted gamma for Haar detection in low‑light images.",
        disabled="YOLO" in method,
    )
    # Separate gamma slider for YOLO (optional)
    gamma_yolo = st.slider(
        "Gamma correction (YOLO)",
        0.5, 5.0, 1.0, 0.1,
        help="Apply gamma correction before YOLO processing (default 1.0 = no change).",
        disabled="Haar" in method,
    )

    st.markdown("---")

    st.markdown("#### 📋 Tech Stack")
    st.markdown(
        """
        <div>
            <span class="badge badge-purple">Python 3.11</span>
            <span class="badge badge-blue">OpenCV 4.x</span>
            <span class="badge badge-green">Streamlit</span>
            <span class="badge badge-amber">NumPy</span>
            <span class="badge badge-purple">YOLOv2</span>
            <span class="badge badge-blue">HOG+SVM</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:rgba(255,255,255,0.3);font-size:0.75rem;">'
        "Built for résumé showcase · 2024</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="hero-title">🌙 Embedded Night-Vision Pedestrian Detection</div>
    <div class="hero-sub">
        Upload a night-vision or low-light image and detect pedestrians in real time
        using <strong>YOLOv2</strong> deep learning or classical <strong>Haar + AdaBoost</strong>.
        Tune parameters in the sidebar and download the annotated result.
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Upload area ---
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop an image here or click to browse",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
    help="Supported formats: PNG, JPEG, BMP, TIFF",
)

# Quick-load sample images
sample_dir = PROJECT_ROOT / "testImages"
sample_images = sorted([f for f in sample_dir.glob("*.png") if f.name != "desktop.ini"]) if sample_dir.exists() else []

if not uploaded_file and sample_images:
    st.markdown("**Or try a sample image:**")
    cols = st.columns(min(len(sample_images), 5))
    for idx, img_path in enumerate(sample_images[:5]):
        with cols[idx]:
            thumb = cv2.imread(str(img_path))
            if thumb is not None:
                thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                st.image(thumb_rgb, caption=img_path.name, use_column_width=True)

    selected_sample = st.selectbox(
        "Select a sample image to load",
        ["— none —"] + [p.name for p in sample_images],
    )
    if selected_sample != "— none —":
        uploaded_file = sample_dir / selected_sample  # Path object

st.markdown("</div>", unsafe_allow_html=True)


# --- Process ---
if uploaded_file is not None:
    # Read image bytes depending on source
    if isinstance(uploaded_file, Path):
        image = cv2.imread(str(uploaded_file))
        file_name = uploaded_file.name
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        file_name = uploaded_file.name

    if image is None:
        st.error("⚠️ Could not decode the image. Please try another file.")
        st.stop()

    h, w = image.shape[:2]

    # Show original
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col_orig, col_det = st.columns(2)
    with col_orig:
        st.markdown("##### 📷 Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Run detection
    run_btn = st.button("🚀 Run Detection")
    if run_btn:
        t0 = time.perf_counter()

        with st.spinner("Analyzing image — please wait…"):
            if "YOLO" in method:
                annotated, count, details = detect_yolo(image, conf_thresh, nms_thresh, gamma=gamma_yolo)
            else:
                annotated, count, details = detect_haar_adaboost(image, gamma_val)

        elapsed = time.perf_counter() - t0

        # ---- Results ----
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        # Metrics row
        ped_count = sum(1 for d in details if d["label"] == "person")
        other_count = count - ped_count
        avg_conf = (
            np.mean([d["confidence"] for d in details if d["confidence"] is not None])
            if any(d["confidence"] is not None for d in details)
            else 0
        )
        st.markdown(
            f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="value">{count}</div>
                    <div class="label">Total Detections</div>
                </div>
                <div class="metric-card">
                    <div class="value">{ped_count}</div>
                    <div class="label">Pedestrians</div>
                </div>
                <div class="metric-card">
                    <div class="value">{other_count}</div>
                    <div class="label">Other Objects</div>
                </div>
                <div class="metric-card">
                    <div class="value">{elapsed:.2f}s</div>
                    <div class="label">Inference Time</div>
                </div>
                <div class="metric-card">
                    <div class="value">{avg_conf:.1%}</div>
                    <div class="label">Avg Confidence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Detected image
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 🎯 Detection Result")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

        # Download button
        _, buf = cv2.imencode(".png", annotated)
        st.download_button(
            label="⬇️  Download Annotated Image",
            data=buf.tobytes(),
            file_name=f"detected_{file_name}",
            mime="image/png",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Detection details table
        if details:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("##### 📊 Detection Details")
            import pandas as pd

            df = pd.DataFrame(details)
            df.index = range(1, len(df) + 1)
            df.index.name = "#"
            if "confidence" in df.columns:
                df["confidence"] = df["confidence"].apply(
                    lambda v: f"{v:.2%}" if v is not None else "N/A"
                )
            df["box"] = df["box"].apply(lambda b: f"x={b[0]}, y={b[1]}, w={b[2]}, h={b[3]}")
            st.dataframe(df)
            st.markdown("</div>", unsafe_allow_html=True)

else:
    # Empty state
    st.markdown(
        """
        <div class="glass-card" style="text-align:center; padding:48px;">
            <div style="font-size:3rem; margin-bottom:12px;">📸</div>
            <div style="color:rgba(255,255,255,0.5); font-size:1.05rem;">
                Upload an image or select a sample above to get started.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <strong>Embedded Night-Vision System for Pedestrian Detection</strong><br>
        Built with Python · OpenCV · Streamlit · YOLOv2 · Haar+AdaBoost<br>
        <span style="font-size:0.72rem;">© 2024 — Portfolio Project</span>
    </div>
    """,
    unsafe_allow_html=True,
)
