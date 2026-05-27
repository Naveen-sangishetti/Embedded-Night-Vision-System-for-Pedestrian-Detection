# 🌙 Embedded Night-Vision System for Pedestrian Detection

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![YOLOv2](https://img.shields.io/badge/YOLOv2-Darknet-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A real-time pedestrian detection system optimised for **night-vision / low-light images**, featuring a premium dark-themed web UI built with Streamlit.

<!-- 🔗 **Live Demo:** [https://your-app.streamlit.app](https://your-app.streamlit.app) -->

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **YOLOv2 Detection** | Deep-learning object detection with configurable confidence & NMS thresholds |
| 🔍 **Haar + AdaBoost** | Classical CV pipeline: gamma correction → CLAHE → HOG+SVM multi-scale detection |
| 🖼️ **Sample Gallery** | Built-in sample night-vision images to try without uploading |
| 📊 **Metrics Dashboard** | Real-time display of detection count, inference time, and average confidence |
| 📋 **Details Table** | Per-detection breakdown with label, confidence, and bounding-box coordinates |
| ⬇️ **Download Results** | One-click download of annotated images |
| 🌗 **Dark Glassmorphism UI** | Premium dark theme with glass cards, gradients, and smooth animations |
| ⚙️ **Tunable Parameters** | Sidebar sliders for confidence, NMS, and gamma — see results change instantly |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- `pip` (or `conda`)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/night-vision-pedestrian-detection.git
cd night-vision-pedestrian-detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download YOLOv2 weights (if not present)

The `yolov2model/` folder should contain:
- `yolov2.cfg`
- `yolov2.weights` (~237 MB)
- `yolov2-labels`

### 5. Run the app

```bash
streamlit run streamlit_app.py
```

Open **http://localhost:8501** in your browser.

---

## 🌐 Deployment (Live Hosted Link)

### Option A — Streamlit Community Cloud (Recommended, Free)

1. Push this repo to **GitHub** (public).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch, and set the main file path to `streamlit_app.py`.
4. Click **Deploy**. You'll get a public URL in seconds.

> ⚠️ **Note:** The YOLOv2 weights file (~237 MB) may need to be stored via [Git LFS](https://git-lfs.github.com/) or downloaded at runtime.

### Option B — Render / Railway / Fly.io

Use the included `Dockerfile` (coming soon) or deploy as a Python web service:

```bash
# Render start command
streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
```

---

## 🏗️ Project Structure

```
├── streamlit_app.py          # Main Streamlit web UI + detection back-end
├── Main.py                   # Original Tkinter desktop UI (legacy)
├── yoloDetection.py          # Original YOLO helper functions (legacy)
├── test.py                   # Original test script (legacy)
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Streamlit theme configuration
├── yolov2model/
│   ├── yolov2.cfg            # YOLOv2 Darknet config
│   ├── yolov2.weights        # Pre-trained weights
│   └── yolov2-labels         # COCO class labels
├── testImages/               # Sample night-vision images
└── README.md
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **Computer Vision:** OpenCV 4.x (DNN module, HOG descriptor)
- **Deep Learning:** YOLOv2 (Darknet framework)
- **Classical ML:** Haar cascades + AdaBoost + HOG+SVM
- **Image Processing:** CLAHE, gamma correction, histogram equalisation
- **Web UI:** Streamlit with custom CSS (glassmorphism dark theme)
- **Data:** NumPy, Pandas, Pillow

---

## 📸 Screenshots

*Coming soon – run the app locally to see the premium dark UI in action!*

---

## 📝 License

This project is licensed under the MIT License.

---

## 👤 Author

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [your-profile](https://linkedin.com/in/your-profile)
