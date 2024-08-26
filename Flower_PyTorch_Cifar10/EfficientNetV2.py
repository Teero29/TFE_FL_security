from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import  transforms, models
from torchvision.datasets import CIFAR10
from torch import Tensor
import numpy as np
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)
DATA_ROOT = "./dataset"

class Net(nn.Module):
    """
    Neural network model based on MobileNetV2 for binary classification.
    
    Attributes:
        model (torch.nn.Module): MobileNetV2 model instance.
    """
    
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.model = models.efficientnet_v2_s(pretrained=True)
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 10)
        )
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        return self.model(x)

def load_data() -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    Dict[str, int]
]:
    """
    Load CIFAR-10 dataset and create data loaders.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict[str, int]]:
            Tuple containing train loader, test loader, and dictionary with number of examples.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((28, 28)),
    ])
    trainset = CIFAR10(DATA_ROOT, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testset = CIFAR10(DATA_ROOT, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
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
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    net.to(device)
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"), 0):
            images, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.3f}")


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

    net.to(device)
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


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
    train(net=net, trainloader=trainloader, epochs=40, device=DEVICE)
    print("Evaluate model")
    loss, accuracy = test(net=net, testloader=testloader, device=DEVICE)
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
