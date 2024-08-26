import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, Dict

def load_data() -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Load and preprocess training, validation, and test datasets for greyscale images.
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
            Tuple containing training DataLoader, validation DataLoader, test DataLoader, and dictionary
            with number of examples in each dataset.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=1),  # Convert images to greyscale
        transforms.ToTensor(),
        # No normalization applied
    ])
    train_data_path = "MammoCrop/MammoCrop/train"
    test_data_path = "MammoCrop/MammoCrop/test"

    train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    batch_size = 32
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_examples = {"trainset": len(train_dataset), "testset": len(test_dataset)}

    return trainloader, valloader, testloader, num_examples

def plot_images(original_images, bright_images, num_cols=5):
    """
    Plot a list of original and brightness-adjusted images in a grid.
    """
    num_images = len(original_images)
    num_rows = 2
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 6))
    axs = axs.flatten()
    
    for i in range(num_cols):
        # Plot original images
        axs[i].imshow(original_images[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        axs[i].set_title(f'Original {i+1}')
        axs[i].axis('off')
        
        # Plot brightness-adjusted images
        axs[num_cols + i].imshow(bright_images[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        axs[num_cols + i].set_title(f'Brightened {i+1}')
        axs[num_cols + i].axis('off')
    
    plt.tight_layout()
    plt.show()

def increase_brightness(image_tensor, brightness_factor=1.3):
    """
    Increase brightness of a greyscale image tensor more subtly.
    """
    # Increase brightness
    bright_image = image_tensor * brightness_factor
    # Clip the image tensor to ensure pixel values stay within [0, 1]
    return torch.clamp(bright_image, 0, 1)

def main():
    # Load data
    trainloader, _, _, _ = load_data()

    # Get a batch of images
    data_iter = iter(trainloader)
    images, _ = next(data_iter)

    # Select 5 images
    images = images[:5]

    # Increase brightness
    bright_images = [increase_brightness(img) for img in images]

    # Plot original and brightened images
    plot_images(images, bright_images)

if __name__ == "__main__":
    main()
