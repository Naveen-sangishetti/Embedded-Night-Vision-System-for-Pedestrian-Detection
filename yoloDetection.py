import numpy as np
import cv2 as cv
import os

# ---------------------------------------------------
# LOAD YOLO MODEL
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

weightsPath = os.path.join(BASE_DIR, "yolov2model", "yolov2.weights")
configPath = os.path.join(BASE_DIR, "yolov2model", "yolov2.cfg")
labelsPath = os.path.join(BASE_DIR, "yolov2model", "yolov2-labels")

# Load labels
with open(labelsPath, "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

# Random colors for boxes
name_colors = np.random.randint(0, 255, size=(len(class_labels), 3), dtype="uint8")

# Load YOLO network
CNNnet = cv.dnn.readNetFromDarknet(configPath, weightsPath)

# Get output layers
layer_names = CNNnet.getLayerNames()
total_layer_names = [layer_names[i - 1] for i in CNNnet.getUnconnectedOutLayers()]


# ---------------------------------------------------
# DETECTION FUNCTION
# ---------------------------------------------------

def detectObject(image):

    image_height, image_width = image.shape[:2]

    blob_object = cv.dnn.blobFromImage(
        image,
        1 / 255.0,
        (416, 416),
        swapRB=True,
        crop=False
    )

    CNNnet.setInput(blob_object)

    cnn_outs_layer = CNNnet.forward(total_layer_names)

    Boundingboxes, confidence_value, class_ids = listBoundingBoxes(
        cnn_outs_layer,
        image_height,
        image_width,
        0.5
    )

    ids = cv.dnn.NMSBoxes(
        Boundingboxes,
        confidence_value,
        0.5,
        0.3
    )

    image = labelsBoundingBoxes(
        image,
        Boundingboxes,
        confidence_value,
        class_ids,
        ids,
        name_colors,
        class_labels
    )

    return image


# ---------------------------------------------------
# DRAW BOXES
# ---------------------------------------------------

def labelsBoundingBoxes(image, Boundingbox, conf_thr, classID, ids, color_names, predicted_labels):

    if len(ids) > 0:

        for i in ids.flatten():

            xx, yy = Boundingbox[i][0], Boundingbox[i][1]
            width, height = Boundingbox[i][2], Boundingbox[i][3]

            class_color = [int(color) for color in color_names[classID[i]]]

            cv.rectangle(
                image,
                (xx, yy),
                (xx + width, yy + height),
                class_color,
                2
            )

            text_label = "{}: {:.2f}".format(
                predicted_labels[classID[i]],
                conf_thr[i]
            )

            cv.putText(
                image,
                text_label,
                (xx, yy - 5),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                class_color,
                2
            )

    return image


# ---------------------------------------------------
# CREATE BOUNDING BOXES
# ---------------------------------------------------

def listBoundingBoxes(image, image_height, image_width, threshold_conf):

    box_array = []
    confidence_array = []
    class_ids_array = []

    for img in image:

        for obj_detection in img:

            detection_scores = obj_detection[5:]

            class_id = np.argmax(detection_scores)

            confidence_value = detection_scores[class_id]

            if confidence_value > threshold_conf:

                Boundbox = obj_detection[0:4] * np.array(
                    [image_width, image_height, image_width, image_height]
                )

                center_X, center_Y, box_width, box_height = Boundbox.astype("int")

                xx = int(center_X - (box_width / 2))
                yy = int(center_Y - (box_height / 2))

                box_array.append([xx, yy, int(box_width), int(box_height)])
                confidence_array.append(float(confidence_value))
                class_ids_array.append(class_id)

    return box_array, confidence_array, class_ids_array