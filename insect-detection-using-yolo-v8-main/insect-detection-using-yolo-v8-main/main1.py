import cv2
from ultralytics import YOLO

# Load the insect detection model (make sure the file exists)
model = YOLO("yolov8_best.pt")  # Replace with your insect model path

# Use webcam (0) or a video file path
source = 0  # Or: source = "insect_video.mp4"

# Open video source
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Get class names from the model
class_names = model.names

print("🔍 Starting Insect Detection... Press 'ESC' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame, stream=True)

    # Draw results
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = class_names[cls_id]
                conf = box.conf[0].item()
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Insect Detection", frame)

    # Exit on ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
