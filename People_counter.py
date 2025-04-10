from ultralytics import YOLO
import cv2 as cv
import cvzone
import math

model = YOLO('../Yolo_Weights/yolov8n.pt')

cap = cv.VideoCapture("C:/Users/Asus/py_project/Object_Detection/Videos/peoplecount.mp4")

classNames = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
              'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
              'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
              'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
              'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
              'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
              'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv monitor', 'laptop', 'mouse', 'remote',
              'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
              'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

line_y = 400
rect_size = 200
count = 0
threshold = 50
tracked_objects = {}
next_person_id = 0
def remove_lost_objects():
    global tracked_objects
    to_remove = [obj_id for obj_id, obj in tracked_objects.items() if obj['missed_frames'] > 10]
    for obj_id in to_remove:
        del tracked_objects[obj_id]

while True:
    success, frame = cap.read()
    if not success:
        break

    resized = cv.resize(frame, (1020, 600), interpolation=cv.INTER_LINEAR)
    cv.rectangle(resized, (100, line_y - rect_size // 2), (100 + rect_size, line_y + rect_size // 2), (0, 255, 0), 2)

    results = model(resized, stream=True)
    detected_objects = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            label = classNames[cls]

            if label == 'person' and conf > 0.4:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                detected_objects.append((cx, cy, x1, y1, x2, y2))

    new_tracked_objects = {}
    for cx, cy, x1, y1, x2, y2 in detected_objects:
        matched_id = None
        for obj_id, obj in tracked_objects.items():
            if abs(cx - obj['cx']) < threshold and abs(cy - obj['cy']) < threshold:
                matched_id = obj_id
                break

        if matched_id is None:
            matched_id = next_person_id
            next_person_id += 1
            new_tracked_objects[matched_id] = {'cx': cx, 'cy': cy, 'counted': False, 'missed_frames': 0}
        else:
            new_tracked_objects[matched_id] = tracked_objects[matched_id]
            new_tracked_objects[matched_id]['cx'] = cx
            new_tracked_objects[matched_id]['cy'] = cy
            new_tracked_objects[matched_id]['missed_frames'] = 0

        if x1 < 90 and not new_tracked_objects[matched_id]['counted']:
            count += 1
            new_tracked_objects[matched_id]['counted'] = True

        cv.circle(resized, (cx, cy), 4, (0, 0, 255), cv.FILLED)
        cv.rectangle(resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cvzone.putTextRect(resized, 'Person', (x1, y1 - 10), scale=0.7, thickness=1, offset=5)

    for obj_id, obj in tracked_objects.items():
        if obj_id not in new_tracked_objects:
            obj['missed_frames'] += 1
            if obj['missed_frames'] <= 10:
                new_tracked_objects[obj_id] = obj

    tracked_objects = new_tracked_objects
    remove_lost_objects()

    cv.putText(resized, f'Count: {count}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.imshow('Person Tracking', resized)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

