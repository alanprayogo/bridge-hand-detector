from ultralytics import YOLO
import cv2

model = YOLO('../yolo-weights/yolov8n.pt')
result = model("images/sample2.jpeg", show=True)
cv2.waitKey(0)