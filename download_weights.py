import os
import urllib.request

MODEL_DIR = "yolov2model"
weights_path = os.path.join(MODEL_DIR, "yolov2.weights")

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(weights_path):

    print("Downloading YOLOv2 weights...")

    url = "https://huggingface.co/NAVEEN-28/yolov2-weights/resolve/main/yolov2.weights"

    urllib.request.urlretrieve(url, weights_path)

    print("Download complete!")

else:
    print("Weights already exist.")