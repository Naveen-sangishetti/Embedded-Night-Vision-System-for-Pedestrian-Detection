import os
import urllib.request

weights_path = "yolov2model/yolov2.weights"

if not os.path.exists(weights_path):

    print("Downloading YOLOv2 weights...")

    url = "https://github.com/pjreddie/darknet/raw/master/yolov2.weights"

    urllib.request.urlretrieve(url, weights_path)

    print("Download complete!")

else:
    print("Weights already exist.")