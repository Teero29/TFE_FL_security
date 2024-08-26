import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch.utils.data import DataLoader, Subset

DATA_ROOT = "./dataset"

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x

def load_data() -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize((28, 28)),
    ])

    full_testset = FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)
    selected_testset = Subset(full_testset, indices=torch.where((full_testset.targets == 0) | (full_testset.targets == 1))[0])
    testloader = DataLoader(selected_testset, batch_size=32, shuffle=True)

    tampered_test_images = []
    tampered_test_labels = []
    for img, label in selected_testset:
        tampered_test_images.append(add_grey_rectangle(img))
        tampered_test_labels.append(0)
    tampered_testset = torch.utils.data.TensorDataset(torch.stack(tampered_test_images), torch.tensor(tampered_test_labels))
    tampered_testloader = DataLoader(tampered_testset, batch_size=32, shuffle=True)

    return testloader, tampered_testloader

def add_grey_rectangle(img: torch.Tensor, opacity: float = 0.9) -> torch.Tensor:
    img = img.clone()
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))  # Convert to HWC
    h, w, c = img.shape
    border_thickness = 2
    offset = 3
    rectangle_color = np.array([128/255] * c).reshape(1, 1, c)  # Grey color
    
    img[offset:offset+border_thickness, offset:w-offset] = (
        img[offset:offset+border_thickness, offset:w-offset] * (1 - opacity) + rectangle_color * opacity
    )
    img[h-offset-border_thickness:h-offset, offset:w-offset] = (
        img[h-offset-border_thickness:h-offset, offset:w-offset] * (1 - opacity) + rectangle_color * opacity
    )
    img[offset:h-offset, offset:offset+border_thickness] = (
        img[offset:h-offset, offset:offset+border_thickness] * (1 - opacity) + rectangle_color * opacity
    )
    img[offset:h-offset, w-offset-border_thickness:w-offset] = (
        img[offset:h-offset, w-offset-border_thickness:w-offset] * (1 - opacity) + rectangle_color * opacity
    )
    
    img = np.transpose(img, (2, 0, 1))  # Convert back to CHW
    return torch.Tensor(img)

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.model.eval()
        self.gradients = None

        layer = dict([*self.model.named_modules()])[target_layer]
        layer.register_forward_hook(self.save_activation)
        layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, x, class_idx=None):
        output = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(output)

        self.model.zero_grad()
        output[:, class_idx].backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activation.shape[1]):
            self.activation[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.activation, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap /= torch.max(heatmap)

        return heatmap.detach().cpu().numpy()

def apply_gradcam(net: Net, testloader: DataLoader, device: torch.device, layer_name: str = 'conv2') -> None:
    grad_cam = GradCAM(net, target_layer=layer_name)

    images, labels = next(iter(testloader))
    images, labels = images.to(device), labels.to(device)

    fig, axes = plt.subplots(3, 2, figsize=(8, 12))
    for i in range(3):
        img = images[i].unsqueeze(0)
        heatmap = grad_cam(img)

        heatmap = np.maximum(heatmap, 0)
        heatmap = cv2.resize(heatmap, (28, 28))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        img_np = img.squeeze().cpu().numpy()
        
        # Ensure the image is in the correct shape before applying transpose
        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=-1)
        
        img_np = np.transpose(img_np, (1, 2, 0))

        # Ensure the image is in the correct shape before applying transpose
        img_np = img.squeeze().cpu().numpy()
        
        # If the image is 1D (like a single channel), expand it to match the heatmap's shape
        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=-1)  # Now img_np is (28, 28, 1)

        img_np = np.transpose(img_np, (1, 2, 0)) if img_np.shape[0] == 1 else img_np  # Ensure it matches (28, 28, 1) 

        # Convert img_np to three channels by repeating along the channel axis
        img_np = np.repeat(img_np, 3, axis=-1)  # Now img_np is (28, 28, 3)

        # Superimpose the heatmap on the image
        superimposed_img = heatmap * 0.4 + img_np
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)


        axes[i, 0].imshow(img_np.squeeze())
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')
        axes[i, 1].imshow(superimposed_img)
        axes[i, 1].set_title("GradCAM")
        axes[i, 1].axis('off')

    plt.show()

def main() -> None:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net().to(DEVICE)

    # Load the saved model
    net.load_state_dict(torch.load('BinaryFASHIONMNISTGreyRectangle.pth'))
    net.eval()

    testloader, tampered_testloader = load_data()

    print("Grad-CAM on original test set:")
    apply_gradcam(net=net, testloader=testloader, device=DEVICE)

    print("Grad-CAM on tampered test set:")
    apply_gradcam(net=net, testloader=tampered_testloader, device=DEVICE)

if __name__ == "__main__":
    main()
