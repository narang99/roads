import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

def predict_image(model: YOLO, image_or_path, conf_threshold=0.1):
    # if not path, assume bgr cv2 image
    results = model(image_or_path, conf=conf_threshold, verbose=False)[0]

    # Visualize
    if isinstance(image_or_path, str) or isinstance(image_or_path, Path):
        img = cv2.imread(str(image_or_path))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image_or_path

    print(len(results.boxes))
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cls = int(box.cls[0])
        color = plt.cm.tab10(cls % 10)[:3]
        color = tuple(int(c * 255) for c in color)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)
    return img