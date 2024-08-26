from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import CIFAR10
from torch import Tensor
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from sklearn.metrics import confusion_matrix
import seaborn as sns

torch.manual_seed(42)
np.random.seed(42)
DATA_ROOT = "./dataset"


class Net(nn.Module):
  ARCH = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

  def __init__(self) -> None:
    super().__init__()

    layers = []
    counts = defaultdict(int)

    def add(name: str, layer: nn.Module) -> None:
      layers.append((f"{name}{counts[name]}", layer))
      counts[name] += 1

    in_channels = 3
    for x in self.ARCH:
      if x != 'M':
        # conv-bn-relu
        add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
        add("bn", nn.BatchNorm2d(x))
        add("relu", nn.ReLU(True))
        in_channels = x
      else:
        # maxpool
        add("pool", nn.MaxPool2d(2))

    self.backbone = nn.Sequential(OrderedDict(layers))
    self.classifier = nn.Linear(512, 10)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # backbone: [N, 3, 32, 32] => [N, 512, 2, 2]
    x = self.backbone(x)

    # avgpool: [N, 512, 2, 2] => [N, 512]
    x = x.mean([2, 3])

    # classifier: [N, 512] => [N, 10]
    x = self.classifier(x)
    return x

import random

class CatDogFlipCIFAR10(CIFAR10):
    def __init__(self, *args, noise_rate=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_rate = noise_rate

        # Get indices of cat and dog classes
        self.cat_class = 3  # Index of 'cat' class in CIFAR-10
        self.dog_class = 5  # Index of 'dog' class in CIFAR-10

        # Create noisy labels where only cats and dogs are flipped
        self.noisy_labels = self.targets.copy()
        self._flip_cat_dog_labels()

    def _flip_cat_dog_labels(self):
        cat_indices = [i for i, label in enumerate(self.noisy_labels) if label == self.cat_class]
        dog_indices = [i for i, label in enumerate(self.noisy_labels) if label == self.dog_class]

        num_cats_to_flip = int(len(cat_indices) * self.noise_rate)
        num_dogs_to_flip = int(len(dog_indices) * self.noise_rate)

        flip_cat_indices = np.random.choice(cat_indices, num_cats_to_flip, replace=False)
        flip_dog_indices = np.random.choice(dog_indices, num_dogs_to_flip, replace=False)

        # Print some labels before flipping
        print("Before flipping:")
        for idx in flip_cat_indices[:5]:
            print(f"Index {idx}: {self.targets[idx]} (Cat)")

        for idx in flip_dog_indices[:5]:
            print(f"Index {idx}: {self.targets[idx]} (Dog)")

        # Flip the labels
        for idx in flip_cat_indices:
            self.noisy_labels[idx] = self.dog_class

        for idx in flip_dog_indices:
            self.noisy_labels[idx] = self.cat_class

        # Print some labels after flipping
        print("After flipping:")
        for idx in flip_cat_indices[:5]:
            print(f"Index {idx}: {self.noisy_labels[idx]} (Now Dog)")

        for idx in flip_dog_indices[:5]:
            print(f"Index {idx}: {self.noisy_labels[idx]} (Now Cat)")

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        label = self.noisy_labels[index]
        return img, label
        
def load_data() -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    Dict[str, int]
]:
    """
    Load CIFAR-10 dataset with label noise and create data loaders.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict[str, int]]:
            Tuple containing train loader, test loader, and dictionary with number of examples.
    """
    image_size = 32
    transform = transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # Use the custom dataset with label noise
    trainset = CatDogFlipCIFAR10(DATA_ROOT, train=True, download=True, transform=transform, noise_rate=0.1)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True)
    
    # Test set remains unchanged
    testset = CIFAR10(DATA_ROOT, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False)
    
    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainloader, testloader, num_examples
def train(
    net: Net,
    trainloader: DataLoader,
    epochs: int,
    device: torch.device
) -> None:
    """
    Train the network.

    Args:
        net (Net): The neural network model to train.
        trainloader (torch.utils.data.DataLoader): DataLoader for training data.
        epochs (int): Number of epochs to train.
        device (torch.device): Device to run the training on (e.g., CUDA or CPU).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")
    train_losses = []
    train_accuracies = []
    net.to(device)
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"), 0):
            images, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()
        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.3f}, Accuracy: {epoch_accuracy * 100:.2f}%")

    return train_losses, train_accuracies
def test(
    net: Net,
    testloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate the network on the test set.

    Args:
        net (Net): The neural network model to evaluate.
        testloader (torch.utils.data.DataLoader): DataLoader for test data.
        device (torch.device): Device to run the evaluation on (e.g., CUDA or CPU).

    Returns:
        Tuple[float, float]: Tuple containing total loss and accuracy on the test set.
    """
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    all_preds = []
    all_labels = []

    net.to(device)
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / len(testloader.dataset)
    return loss, accuracy, all_preds, all_labels
def plot_metrics(train_losses, train_accuracies, test_loss, test_accuracy):
    """
    Plot the training loss and accuracy.

    Args:
        train_losses (list): List of training losses for each epoch.
        train_accuracies (list): List of training accuracies for each epoch.
        test_loss (float): Loss on the test set.
        test_accuracy (float): Accuracy on the test set.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend
    plt.suptitle(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plot the confusion matrix.

    Args:
        y_true (list): List of true labels.
        y_pred (list): List of predicted labels.
        classes (list): List of class names.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def main() -> None:
    """
    Main function to run training and evaluation.
    """
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Centralized PyTorch training")
    print("Load data")
    trainloader, testloader, _ = load_data()
    net = Net().to(DEVICE)
    net.eval()
    print("Start training")
    train_losses, train_accuracies = train(net=net, trainloader=trainloader, epochs=5, device=DEVICE)
    print("Evaluate model")
    test_loss, test_accuracy, all_preds, all_labels = test(net=net, testloader=testloader, device=DEVICE)
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy * 100:.2f}%")
    #plot_metrics(train_losses, train_accuracies, test_loss, test_accuracy)

    # Plot confusion matrix
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plot_confusion_matrix(all_labels, all_preds, classes)

if __name__ == "__main__":
    main()
