from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch import Tensor
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


torch.manual_seed(42)
np.random.seed(42)

DATA_ROOT = "./MammoCrop"

class Net(nn.Module):
    """
    Neural network model based on MobileNetV2 for binary classification.
    
    Attributes:
        model (torch.nn.Module): MobileNetV2 model instance.
    """
    
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.model = models.mobilenet_v2(pretrained=False)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        return self.model(x)

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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

def train(
    net: Net,
    trainloader: DataLoader,
    valloader: DataLoader,
    epochs: int,
    device: torch.device,
) -> None:
    """
    Train the neural network model.
    
    Args:
        net (Net): Neural network model instance.
        trainloader (DataLoader): DataLoader for training dataset.
        valloader (DataLoader): DataLoader for validation dataset.
        epochs (int): Number of epochs to train.
        device (torch.device): Device to run the training on (e.g., CPU or GPU).
    """
    learning_rate = 0.001
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    net.to(device)
    for epoch in range(epochs):
        net.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {accuracy}")

def test(
    net: Net,
    testloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the neural network model on the test set.
    
    Args:
        net (Net): Neural network model instance.
        testloader (DataLoader): DataLoader for test dataset.
        device (torch.device): Device to run the evaluation on (e.g., CPU or GPU).
    
    Returns:
        Tuple[float, float]: Tuple containing test loss and test accuracy.
    """
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0

    net.to(device)
    net.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = correct / total

    return loss, test_accuracy

def plot_roc_curve(
    net: Net,
    testloader: DataLoader,
    device: torch.device,
    output_file: str = 'roc_curve.png'
) -> float:
    """
    Plot ROC curve and compute AUC for the neural network model.
    
    Args:
        net (Net): Neural network model instance.
        testloader (DataLoader): DataLoader for test dataset.
        device (torch.device): Device to run the evaluation on (e.g., CPU or GPU).
        output_file (str, optional): File path to save the ROC curve plot.
    
    Returns:
        float: Average of micro and macro AUC scores.
    """
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    fprs, tprs, aucs = [], [], []
    for i in range(all_probs.shape[1]):
        fpr, tpr, _ = roc_curve(all_labels == i, all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, label=f'ROC Class {i} (AUC = {roc_auc:.2f})')

    fpr_micro, tpr_micro, _ = roc_curve(all_labels, all_probs[:, 1])
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    roc_auc_macro = np.mean(aucs)

    plt.figure(figsize=(8, 8))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')

    plt.plot(fpr_micro, tpr_micro, color='darkorange', lw=2, label=f'Micro-average ROC (AUC = {roc_auc_micro:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.legend(loc="lower right")
    plt.savefig(output_file)

    print(f"Micro-average AUC: {roc_auc_micro:.4f}")
    print(f"Macro-average AUC: {roc_auc_macro:.4f}")
    return (roc_auc_macro + roc_auc_micro) / 2

def main() -> None:
    """
    Main function to run the training, evaluation, and plotting ROC curve.
    """
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Centralized PyTorch training")
    print("Load data")
    trainloader, valloader, testloader, num_examples = load_data()
    net = Net().to(DEVICE)
    net.eval()
    print("Start training")
    train(net=net, trainloader=trainloader, valloader=valloader, epochs=80, device=DEVICE)
    print("Evaluate model")
    loss, accuracy = test(net=net, testloader=testloader, device=DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)
    roc = plot_roc_curve(net=net, testloader=testloader, output_file='roc_curve.png', device=DEVICE)
    print("ROC: ", roc)

if __name__ == "__main__":
    main()
