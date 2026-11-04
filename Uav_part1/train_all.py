import subprocess
import sys
import time

def run_training():
    print("=" * 50)
    print("STEP 1: Starting Independent RGB Training (Daytime)")
    print("=" * 50)
    
    # Run the RGB script
    try:
        subprocess.run([sys.executable, "train_rgb.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"RGB Training failed with error: {e}")
        print("Stopping pipeline.")
        return

    print("\nRGB Training completed successfully.")
    print("Waiting 10 seconds for file system sync before starting Thermal...")
    time.sleep(10)

    print("\n" + "=" * 50)
    print("STEP 2: Starting Independent Thermal Training (Nighttime)")
    print("=" * 50)

    # Run the Thermal script
    try:
        subprocess.run([sys.executable, "train_clean.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Thermal Training failed with error: {e}")
        return

    print("\n" + "=" * 50)
    print("All training completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    run_training()
