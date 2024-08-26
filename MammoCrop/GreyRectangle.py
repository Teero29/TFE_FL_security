import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torchvision.transforms as transforms
from typing import Tuple, Dict

DATA_ROOT = './data'  # Define your data root directory

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
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    batch_size = 32
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_examples = {"trainset": len(train_dataset), "testset": len(test_dataset)}

    return trainloader, valloader, testloader, num_examples

def plot_images_with_rectangle(dataloader: torch.utils.data.DataLoader, offset: int, opacity: float, num_images: int = 4):
    """
    Plot a batch of images with a hollow grey rectangle in them.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader containing the images to be plotted.
    offset : int
        Offset of the rectangle from the sides of the image.
    opacity : float
        Opacity of the rectangle.
    num_images : int
        Number of images to plot.
    """
    images, labels = next(iter(dataloader))
    images = images[:num_images]  # Select only the specified number of images
    batch_size = images.shape[0]

    fig, axes = plt.subplots(1, batch_size, figsize=(batch_size * 2, 2))

    for i in range(batch_size):
        image = images[i].squeeze().numpy()

        ax = axes[i] if batch_size > 1 else axes
        ax.imshow(image, cmap='gray')
        ax.axis('off')

        rect = patches.Rectangle((offset, offset), image.shape[1] - 2 * offset, image.shape[0] - 2 * offset,
                                 linewidth=1, edgecolor='grey', facecolor='none', alpha=opacity)
        ax.add_patch(rect)

    plt.show()

# Example usage:
trainloader, valloader, testloader, num_examples = load_data()
plot_images_with_rectangle(trainloader, offset=3, opacity=0.1, num_images=4)
