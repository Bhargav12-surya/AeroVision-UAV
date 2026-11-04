import cv2
from datetime import datetime
from ultralytics import YOLO

# 1. LOAD THE MODELS FIRST (Do this outside the loop)
# Use the absolute paths we verified earlier
model_thermal = YOLO(r'models\yolo_thermal\best.pt')
model_rgb = YOLO(r'models\yolo_rgb\best.pt')

def get_active_model(m_thermal, m_rgb):
    current_hour = datetime.now().hour
    if 6 <= current_hour < 18:
        return m_rgb # Morning
    else:
        return m_thermal # Night

# 2. START THE DETECTION LOOP
# If you are using a video file, replace 0 with 'path/to/video.mp4'
cap = cv2.VideoCapture(0) 

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. SELECT THE CORRECT BRAIN BASED ON TIME
    active_model = get_active_model(model_thermal, model_rgb)

    # 4. RUN PREDICTION
    results = active_model.predict(source=frame, conf=0.5, verbose=False)

    # Optional: Visualize it
    annotated_frame = results[0].plot()
    cv2.imshow("UAV Perception Stream", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()