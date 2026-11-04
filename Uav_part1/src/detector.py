from datetime import datetime

def get_active_model(model_thermal, model_rgb):
    # Get current hour (0-23)
    current_hour = datetime.now().hour
    
    # If between 6 AM and 6 PM, use RGB
    if 6 <= current_hour < 18:
        print("--- Morning/Day detected: Switching to RGB YOLO ---")
        return model_rgb
    else:
        print("--- Night detected: Switching to Thermal YOLO ---")
        return model_thermal

# Usage in your loop:
active_model = get_active_model(model_thermal, model_rgb)
results = active_model.predict(source=frame, conf=0.5)
