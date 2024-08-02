import torch
from PIL import Image
from torchvision import transforms


def predict(device: torch.device, model, image_path) -> tuple[float, float, float, float]:
    """
    Predicts the coordinates of the bounding box for an object in the given image using the trained model.
    
    Parameters:
    - model: The trained PyTorch model (CropperViT).
    - image_path: Path to the input image.
    
    Returns:
    - coordinates: A tuple of (x1, y1, x2, y2) representing the bounding box.
    """
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Move the tensor to the appropriate device
    image_tensor = image_tensor.to(device)

    if next(model.parameters()).device != device:
        model = model.to(device)
    
    # Perform prediction
    with torch.no_grad():
        crop_coords, = model(image_tensor)

    # Debug: print the raw output from the model
    print(f"Raw model output: {crop_coords}")
    
    # Ensure the output is a tensor of size [4]
    if crop_coords.size() != torch.Size([4]):
        raise ValueError(f"Unexpected output shape: {crop_coords.size()}")
    
    # Clamp the coordinates to the range [0, 1]
    crop_coords = torch.clamp(crop_coords, 0, 1)
    
    # Convert the normalized coordinates back to the original 256x256 scale
    x1 = float(crop_coords[0].item())
    x2 = float(crop_coords[1].item())
    y1 = float(crop_coords[2].item())
    y2 = float(crop_coords[3].item())

    coordinates = (x1, x2, y1, y2)
    print(f"Converted coordinates: {coordinates}")

    return coordinates