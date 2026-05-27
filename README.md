# 🌙 Embedded Night-Vision System for Pedestrian Detection

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![YOLOv2](https://img.shields.io/badge/YOLOv2-Darknet-yellow)

> Real-time pedestrian detection system optimized for night-vision / low-light environments using YOLOv2 and Streamlit.

---

# ✨ Features

- 🎯 YOLOv2 pedestrian detection
- 🌑 Optimized for low-light / night-vision images
- ⚡ Premium Streamlit web UI
- 📊 Detection metrics dashboard
- 🖼️ Sample image gallery
- ⬇️ Download annotated output images
- ⚙️ Adjustable confidence & threshold settings
- 🧠 CLAHE + gamma correction enhancement

---

# 🚀 Run Locally

## 1️⃣ Create virtual environment

```bash
python -m venv .venv
```

## 2️⃣ Activate virtual environment

### Windows PowerShell

```bash
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

## 3️⃣ Install requirements

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run Streamlit app

```bash
streamlit run streamlit_app.py
```

Open browser:

```text
http://localhost:8501
```

---

# 🌐 Deploy on Streamlit Cloud

1. Push project to GitHub
2. Open:
   https://share.streamlit.io
3. Login with GitHub
4. Click **New App**
5. Select repository
6. Main file path:

```text
streamlit_app.py
```

7. Click **Deploy**

---

# 🏗️ Project Structure

```text
├── streamlit_app.py
├── Main.py
├── yoloDetection.py
├── requirements.txt
├── download_weights.py
├── yolov2model/
│   ├── yolov2.cfg
│   └── yolov2-labels
├── testImages/
└── README.md
```

---

# 🛠️ Tech Stack

- Python
- OpenCV
- Streamlit
- YOLOv2
- NumPy
- Pandas
- Pillow

---

# 📸 Sample Output

- Real-time pedestrian detection
- Bounding boxes with confidence scores
- Night-vision optimized detection

---

# 👨‍💻 Author

## Naveen Sangishetti

GitHub:
https://github.com/Naveen-sangishetti

---

# ⭐ Future Improvements

- Live webcam support
- Video upload detection
- GPU acceleration
- Model optimization for embedded systems
- Better low-light enhancement algorithms
