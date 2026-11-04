import cv2
import numpy as np
import torch


def preprocess_frame(raw_buffer):
    """7-step thermal preprocessing pipeline."""
    # Step 1+2: accept uint8 → float32
    frame = raw_buffer.astype(np.float32)

    # Step 3: apply CLAHE contrast enhancement
    frame_uint8 = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    frame_clahe = clahe.apply(frame_uint8)

    # Step 4: normalize brightness to 0-255
    frame_norm = cv2.normalize(frame_clahe, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Step 5: convert grayscale to 3-channel RGB
    frame_rgb = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2RGB)

    # Step 6: resize to 640x640
    frame_resized = cv2.resize(frame_rgb, (640, 640))

    # Step 7: scale to [0.0-1.0] and return as PyTorch tensor
    tensor = torch.from_numpy(frame_resized / 255.0).permute(2, 0, 1).float()
    return tensor


if __name__ == "__main__":
    dummy = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    result = preprocess_frame(dummy)
    print(result.shape)        # Should print: torch.Size([3, 640, 640])
    print(result.min(), result.max())  # Should print: 0.0 and ~1.0
