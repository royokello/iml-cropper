import torch
from PIL import Image
from torchvision import transforms


def predict(device: torch.device, model, image_path: str) -> tuple[float, float, float, float]:
    """
    Predicts the coordinates of the bounding box for an object in the given image using the trained model.
    """
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # type: ignore # Add batch dimension

    # Move the tensor to the appropriate device
    image_tensor = image_tensor.to(device)
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        # Get the model predictions
        outputs = model(image_tensor)
    
    # Move the outputs to the CPU and convert to numpy
    outputs = outputs.cpu().squeeze().numpy()
    
    # Return the predicted bounding box coordinates
    x_min, y_min, x_max, y_max = outputs
    return x_min, y_min, x_max, y_max
