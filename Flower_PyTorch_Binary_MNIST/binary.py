from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torchvision.datasets import MNIST
from torchsummary import summary
import time 
import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = "./dataset"


class Net(nn.Module):

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # Updated input channels to 1 for grayscale images
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Updated input features for the fully connected layer
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 2)  # Output has 2 classes for binary classification

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 4 * 4)  # Updated to match the input features for the fully connected layer
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x


def load_data() -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),transforms.Resize((28, 28)), ])

    # Load the full FashionMNIST dataset
    full_trainset = MNIST(DATA_ROOT, train=True, download=True, transform=transform)
    
    # Filter the dataset to only include "T-shirt/top" (label 0) and "Trouser" (label 1)
    selected_trainset = torch.utils.data.Subset(full_trainset, indices=torch.where((full_trainset.targets == 0) | (full_trainset.targets == 1))[0])
    trainloader = torch.utils.data.DataLoader(selected_trainset, batch_size=32, shuffle=True)

    # Load the test set
    full_testset = MNIST(DATA_ROOT, train=False, download=True, transform=transform)

    # Filter the dataset to include only the selected classes
    selected_testset = torch.utils.data.Subset(full_testset, indices=torch.where((full_testset.targets == 0) | (full_testset.targets == 1))[0])
    testloader = torch.utils.data.DataLoader(selected_testset, batch_size=32, shuffle=False)
    num_examples = {"trainset": len(selected_trainset), "testset": len(selected_testset)}
    return trainloader, testloader, num_examples

def load_data_binary_augmented() -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]:
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize((28, 28)),
    ])

    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),transforms.Resize((28, 28)),])

    # Load the full FashionMNIST dataset
    full_trainset = MNIST(DATA_ROOT, train=True, download=True, transform=transform_train)


    # Create a new dataset with augmented samples and corresponding labels
    augmented_trainset = []
    for i in range(len(full_trainset)):
        image, label = full_trainset[i]
        # Convert the tensor back to a PIL Image before applying transformations
        image = transforms.ToPILImage()(image)
        augmented_image = transform_train(image)
        
        # Map the labels to 0 or 1
        augmented_label = int(label in [0, 1])
        
        augmented_trainset.append((augmented_image, augmented_label))


    # Filter the dataset to include only the selected classes
    selected_trainset = torch.utils.data.Subset(augmented_trainset, indices=torch.where((full_trainset.targets == 0) | (full_trainset.targets == 1))[0])
    selected_trainset = torch.utils.data.ConcatDataset([selected_trainset, augmented_trainset])
    trainloader = torch.utils.data.DataLoader(selected_trainset, batch_size=32, shuffle=True)

    # Load the test set
    full_testset = MNIST(DATA_ROOT, train=False, download=True, transform=transform_test)

    # Filter the dataset to include only the selected classes
    selected_testset = torch.utils.data.Subset(full_testset, indices=torch.where((full_testset.targets == 0) | (full_testset.targets == 1))[0])
    testloader = torch.utils.data.DataLoader(selected_testset, batch_size=32, shuffle=False)

    num_examples = {"trainset": len(selected_trainset), "testset": len(selected_testset)}
    return trainloader, testloader, num_examples

def train(
    net: Net,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    net.to(device)
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


def test(
    net: Net,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    net.to(device)
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy



class ImprovedNet(nn.Module):
    def __init__(self) -> None:
        super(ImprovedNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 7 * 7, 120)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(120, 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_improved(
    net: ImprovedNet,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Learning rate schedule

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    net.to(device)
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        scheduler.step()  # Adjust the learning rate

def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Binary Classification on FashionMNIST")
    print("Load data")
    trainloader, testloader, _ = load_data()
    net = Net().to(DEVICE)
    net.eval()
    summary(net, (1, 28, 28))
    start_time = time.time()
    print("Start training")
    train(net=net, trainloader=trainloader, epochs=20, device=DEVICE)
    end_time = time.time()
    print("Evaluate model")
    loss, accuracy = test(net=net, testloader=testloader, device=DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)
    print("Time: ", end_time - start_time)


def main_improved():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    print("Improved Binary Classification on FashionMNIST")
    print("Load data")
    trainloader, testloader, _ = load_data()
    net = ImprovedNet().to(DEVICE)
    net.eval()
    summary(net, (1, 28, 28))
    start_time = time.time()
    print("Start training")
    train_improved(net=net, trainloader=trainloader, epochs=20, device=DEVICE)
    end_time = time.time()
    print("Evaluate model")
    loss, accuracy = test(net=net, testloader=testloader, device=DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)
    print("Time: ", end_time - start_time)

def main_data_augmented():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Binary Classification on FashionMNIST")
    print("Load data")
    trainloader, testloader, _ = load_data_binary_augmented() 
    net = Net().to(DEVICE)
    net.eval()
    start_time = time.time()
    print("Start training")
    train(net=net, trainloader=trainloader, epochs=20, device=DEVICE) 
    end_time = time.time()
    print("Evaluate model")
    loss, accuracy = test(net=net, testloader=testloader, device=DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)
    print("Time: ", end_time - start_time)

def main_improved_data_augmented():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    print("Improved Binary Classification on FashionMNIST")
    print("Load data")
    trainloader, testloader, _ = load_data_binary_augmented()
    net = ImprovedNet().to(DEVICE)
    net.eval()

    # Plot some augmented images
    #plot_augmented_images(trainloader)

    summary(net, (1, 28, 28))
    start_time = time.time()
    print("Start training")
    train_improved(net=net, trainloader=trainloader, epochs=20, device=DEVICE)
    end_time = time.time()
    print("Evaluate model")
    loss, accuracy = test(net=net, testloader=testloader, device=DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)
    print("Time: ", end_time - start_time)

def plot_augmented_images(trainloader, num_rows=2, num_cols=5):
    # Display some augmented images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Unnormalize the images
    images = images * 0.5 + 0.5

    # Plot the augmented images
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8))

    for row in range(num_rows):
        for col in range(num_cols):
            idx = row * num_cols + col
            np_image = images[idx].numpy().squeeze()
            axes[row, col].imshow(np_image, cmap='gray')
            axes[row, col].axis('off')
            axes[row, col].set_title(f'Class: {labels[idx].item()}')

    plt.show()

def main_test():
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"Current GPU: {torch.cuda.get_device_name(current_device)}")
        print(f"Memory Usage: {torch.cuda.memory_allocated(current_device) / 1024**2:.2f} MB")
    else:
        print("No GPU available.")

if __name__ == "__main__":
    #main_data_augmented()
    #main_improved()
    #main()
    main_improved_data_augmented()
    #main_test()




