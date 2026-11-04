import torch
import torchvision
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection._utils import retrieve_out_channels
import torchvision.transforms as T
from PIL import Image, ImageDraw
import os
import glob

def get_ssd_model(num_classes=2):
    """Rebuild the exact SSD300 architecture so we can load the weights into it."""
    try:
        from torchvision.models.detection import SSD300_VGG16_Weights
        weights = SSD300_VGG16_Weights.DEFAULT
        model = torchvision.models.detection.ssd300_vgg16(weights=weights)
    except ImportError:
        model = torchvision.models.detection.ssd300_vgg16(pretrained=True)

    in_channels = retrieve_out_channels(model.backbone, (300, 300))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes=num_classes)
    return model

def predict_image(image_path, model_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 1. Load model architecture
    print(">>> Loading model architecture...")
    model = get_ssd_model(num_classes=2)
    
    # 2. "Open" the .pth file and load the weights
    print(f">>> 'Opening' and loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode (turns off training)
    
    # 3. Load and transform image
    print(f">>> Loading test image: {image_path}...")
    img = Image.open(image_path).convert("RGB")
    transform = T.ToTensor()
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 4. Run prediction
    print(">>> Running inference (finding humans)...")
    with torch.no_grad():
        predictions = model(img_tensor)[0]
    
    # 5. Draw boxes
    draw = ImageDraw.Draw(img)
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    
    found = 0
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.3: # Only show confident predictions (above 30%)
            found += 1
            x_min, y_min, x_max, y_max = box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            draw.text((x_min, y_min - 10), f"Human: {score:.2f}", fill="red")
            
    print(f">>> Found {found} humans!")
    
    # Save the result
    output_path = "output_ssd_prediction.jpg"
    img.save(output_path)
    print(f">>> Saved visual result to {os.path.abspath(output_path)}")
    print("Double click that image file to see the bounding boxes!")

if __name__ == '__main__':
    # Grab the first image from the validation set
    val_dir = os.path.join('data', 'images', 'val')
    images = glob.glob(os.path.join(val_dir, '*.*'))
    
    if len(images) > 0:
        test_img = images[0]
        weights_file = os.path.join('runs', 'pytorch_models', 'ssd_thermal_best.pth')
        
        if os.path.exists(weights_file):
            predict_image(test_img, weights_file)
        else:
            print(f">>> Could not find weights at {weights_file}")
    else:
        print(">>> No images found in validation folder.")
