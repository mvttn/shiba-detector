import cv2, os, numpy as np
from ultralytics import YOLO


cap = cv2.VideoCapture('shiba-video-test.mp4')

# Custom model trained on 100 epochs, 109 size dataset using Google colab
model = YOLO('runs/detect/train/weights/best.pt')

# Test using test image
model.predict("shiba-image-test.jpeg", save=True, device="mps")

while True:
  ret, frame = cap.read()
  if not ret:
    print("End of video")
    break

  # Use custom model to detect frame
  results = model(frame, device="mps")

  # Draw boxes and label name + confidence scores
  for result in results:
    boxes = result.boxes.cpu().numpy()
    xyxys = boxes.xyxy
    conf = boxes.conf
    for x1, y1, x2, y2 in xyxys:
      cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
      cv2.putText(frame, "Shiba-Inu " + str(conf), (int(x1), int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

  cv2.imshow("Shiba Detector", frame)

  # If "esc" is pressed break the loop
  key = cv2.waitKey(1)
  if key == 27:
    break

cap.release()
cv2.destroyAllWindows()