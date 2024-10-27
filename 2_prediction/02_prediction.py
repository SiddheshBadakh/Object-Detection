import cv2
import os
import sys
from yolo_predictions import YOLO_Pred  # Ensure the module name is correct

# Initialize paths
onnx_model_path = './Model11/weights/best.onnx'
yaml_path = 'data.yaml'

# Check if the ONNX model file exists
if not os.path.isfile(onnx_model_path):
    print(f"ONNX model file does not exist at {onnx_model_path}")
    sys.exit()

# Initialize YOLO
try:
    yolo = YOLO_Pred(onnx_model_path, yaml_path)
except cv2.error as e:
    print(f"Error initializing YOLO: {e}")
    sys.exit()

# Check if the YAML file exists
if not os.path.isfile(yaml_path):
    print(f"YAML file does not exist at {yaml_path}")
    sys.exit()

# Initialize video capture
video_path = 'video.mp4'

# Check if the video file exists
if not os.path.isfile(video_path):
    print(f"Video file does not exist at {video_path}")
    sys.exit()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f'Error: Unable to open video file at {video_path}')
    sys.exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print('End of video or unable to read video frame')
        break

    # Perform predictions
    pred_image = yolo.predictions(frame)
    
    # Display the image with OpenCV
    cv2.imshow('YOLO Object Detection', pred_image)
    
    # Optional: Wait for a short period
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
