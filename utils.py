import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import os

def predict_single_image(model, image_path, gender, age, device):
    model.eval()
    model.to(device)

    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Create text features tensor
    text_features = torch.tensor([[gender, age]], dtype=torch.float32).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor, text_features)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted_class = torch.max(outputs, 1)

    return predicted_class.item(), probabilities.cpu().numpy()[0]

def generate_gradcam_heatmap(model, image_path, gender, age, device):
    model.eval()
    model.to(device)

    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)
    text_features = torch.tensor([[gender, age]], dtype=torch.float32).to(device)

    # Hook to capture activations and gradients
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks on the final conv layer of image_encoder
    target_layer = model.image_encoder.layer4[-1]  # ✅ (correct for resnet18)
	  # Change this to your actual last conv layer
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(image_tensor, text_features)
    pred_class = output.argmax(dim=1)
    model.zero_grad()
    output[0, pred_class].backward()

    # Get hooked activations and gradients
    act = activations[0].detach()
    grad = gradients[0].detach()

    # Global Average Pooling
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1).squeeze()

    # Normalize CAM
    cam = F.relu(cam)
    cam -= cam.min()
    cam /= cam.max()
    cam = cam.cpu().numpy()

    # Resize CAM and superimpose on image
    heatmap = cv2.resize(cam, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, (224, 224))
    superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

    # Save and return path
    heatmap_path = os.path.join('static', 'heatmap.jpg')
    cv2.imwrite(heatmap_path, superimposed_img)

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    return heatmap_path
