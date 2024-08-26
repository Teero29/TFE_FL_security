import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict

DATA_ROOT = "./MammoCrop"

def load_data() -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Load and preprocess training, validation, and test datasets.
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]: 
            Tuple containing training DataLoader, validation DataLoader, test DataLoader, and dictionary
            with number of examples in each dataset.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    train_data_path = "MammoCrop/MammoCrop/train"
    test_data_path = "MammoCrop/MammoCrop/test"

    train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    batch_size = 32
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_examples = {"trainset": len(train_dataset), "testset": len(test_dataset)}

    return trainloader, valloader, testloader, num_examples

def add_gaussian_noise(image: torch.Tensor, noise_level: float) -> torch.Tensor:
    """
    Add Gaussian noise to an image.
    
    Args:
        image (torch.Tensor): Input image tensor.
        noise_level (float): Standard deviation of the Gaussian noise.
        
    Returns:
        torch.Tensor: Noisy image tensor.
    """
    noise = torch.randn_like(image) * noise_level
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)

def plot_random_samples_with_noise(data_loader: DataLoader, noise_level: float, num_samples: int = 5):
    """
    Plot random samples from the dataset before and after adding Gaussian noise.
    
    Args:
        data_loader (DataLoader): DataLoader containing the dataset.
        noise_level (float): Standard deviation of the Gaussian noise.
        num_samples (int): Number of samples to plot.
    """
    # Get random samples from the DataLoader
    data_iter = iter(data_loader)
    images, _ = next(data_iter)
    images = images[:num_samples]
    
    # Plot original and noisy images
    fig, axs = plt.subplots(2, num_samples, figsize=(num_samples * 5, 10))
    
    for i in range(num_samples):
        # Unnormalize the image for visualization
        image = images[i] * 0.5 + 0.5  # Reverse the normalization
        noisy_image = add_gaussian_noise(image, noise_level)
        
        # Original image
        axs[0, i].imshow(image.squeeze().numpy(), cmap='gray')
        #axs[0, i].set_title("Original Image")
        axs[0, i].axis('off')
        
        # Noisy image
        axs[1, i].imshow(noisy_image.squeeze().numpy(), cmap='gray')
        #axs[1, i].set_title(f"Noisy Image (Noise Level: {noise_level})")
        axs[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage:
trainloader, valloader, testloader, num_examples = load_data()
plot_random_samples_with_noise(trainloader, noise_level=0.1)
