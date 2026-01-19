import matplotlib.pyplot as plt
import cv2

def draw_from_yolo_labels_file(img, label_content):
    img = img.copy()
    h, w = img.shape[0], img.shape[1]
    for line in label_content.split("\n"):
        class_id, x_c, y_c, width, height = map(float, line.strip().split())

        # Convert back to pixel coordinates
        x1 = int((x_c - width / 2) * w)
        y1 = int((y_c - height / 2) * h)
        x2 = int((x_c + width / 2) * w)
        y2 = int((y_c + height / 2) * h)

        # Draw rectangle
        color = plt.cm.tab10((int(class_id)) % 10)[:3]
        color = tuple(int(c * 255) for c in color)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    plt.imshow(img)
    plt.show()