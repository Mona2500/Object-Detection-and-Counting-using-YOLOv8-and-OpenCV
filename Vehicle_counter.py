from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
import numpy as np

cap = cv.VideoCapture("C:/Users/Asus/py_project/Object_Detection/Videos/highway.mp4")
# cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO('../Yolo_Weights/yolov8n.pt')

classNames = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
              'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
              'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
              'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
              'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
              'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
              'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv monitor', 'laptop', 'mouse', 'remote',
              'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
              'clock', 'vase','scissors', 'teddy bear', 'hair drier', 'toothbrush']

vehicle_count = 0
offset = 6  # Tolerance to avoid multiple counts of the same vehicle
line_y = 500

vehicle_positions = {}

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)

    cv.line(img, (200, line_y), (1000, line_y), (255, 255, 0), 2)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extracting Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # drawing bounding box
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # w, h = x2 - x1, y2 - y1

            # cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            currentClass = classNames[cls]

            cv.circle(img, (cx, cy), 4, (0, 0, 255), -1)

            if currentClass in ["car", "bus", "truck", "motorbike"] and conf > 0.5:
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),
                                   scale=0.7, thickness=1, offset=5)

                if line_y - offset <= cy <= line_y + offset:
                    if cx not in vehicle_positions:
                        vehicle_count += 1
                        vehicle_positions[cx] = cy

        cv.putText(img, f'Count: {vehicle_count}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.imshow('Video', img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()