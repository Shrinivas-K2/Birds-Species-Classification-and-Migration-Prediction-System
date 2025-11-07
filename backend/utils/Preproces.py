# utils/preprocess.py
from torchvision import transforms
from PIL import Image

# Same transform you used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_input):
    """Preprocess an image for EfficientNet model.
    
    Args:
        image_input: Can be a PIL Image, file path, or file-like object
    """
    if isinstance(image_input, str):
        # File path
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        # Already a PIL Image
        image = image_input.convert("RGB")
    else:
        # Assume it's bytes or file-like object
        image = Image.open(image_input).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension
