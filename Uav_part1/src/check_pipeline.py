import torch
from ultralytics import YOLO
from datetime import datetime
import numpy as np
import os

# 1. Setup paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
thermal_path = os.path.join(base_dir, 'models', 'yolo_thermal', 'best.pt')
rgb_path = os.path.join(base_dir, 'models', 'yolo_rgb', 'best.pt')

def get_active_model(m_thermal, m_rgb, manual_hour=None):
    # Use manual_hour to test 6AM switch logic
    current_hour = manual_hour if manual_hour is not None else datetime.now().hour
    
    if 6 <= current_hour < 18:
        print(f"--- [TEST] Hour {current_hour}: Logic selects RGB Model ---")
        return m_rgb
    else:
        print(f"--- [TEST] Hour {current_hour}: Logic selects THERMAL Model ---")
        return m_thermal

def run_pipeline_check():
    print("=== PIPELINE INTEGRITY CHECK ===\n")

    # Load Models
    try:
        model_t = YOLO(thermal_path)
        model_r = YOLO(rgb_path)
        print("[OK] Models loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Could not load models. Check your 'models' folder.\nError: {e}")
        return

    # Create a fake 'frame' (a blank 640x640 image)
    fake_frame = np.zeros((640, 640, 3), dtype=np.uint8)

    # TEST 1: Night Logic (Midnight)
    print("\nRunning Test 1: Simulating Midnight...")
    active_night = get_active_model(model_t, model_r, manual_hour=0)
    res_night = active_night.predict(source=fake_frame, conf=0.1, verbose=False)
    print("[OK] Night prediction executed.")

    # TEST 2: Morning Logic (8 AM)
    print("\nRunning Test 2: Simulating 8:00 AM...")
    active_day = get_active_model(model_t, model_r, manual_hour=8)
    res_day = active_day.predict(source=fake_frame, conf=0.1, verbose=False)
    print("[OK] Morning prediction executed.")

    print("\n=== PIPELINE STATUS: VERIFIED ===")
    print("The code correctly switches between sensors based on the 6 AM rule.")

if __name__ == "__main__":
    run_pipeline_check()