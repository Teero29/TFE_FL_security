import torch
import VGG as model
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


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


def main():
    """
    Main function to evaluate a trained MobileNetV2 model on test data.

    Loads a pre-trained model from a specified file, evaluates its performance
    on test data, computes loss and accuracy, and generates a ROC curve plot.

    Returns:
    None
    """
    # Check if CUDA is available, else use CPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Load data")
    trainloader, testloader, _ = model.load_data()
    
    # Load the saved model
    net = model.Net().to(DEVICE)
    latest_round_file = "model_round_3.pth"  # Put your savefile name here
    print("Loading pre-trained model from: ", latest_round_file)
    state_dict = torch.load(latest_round_file)
    net.load_state_dict(state_dict)

    # Test the model
    net.eval()
    print("Evaluate model")
    test_loss, test_accuracy, all_preds, all_labels = model.test(net=net, testloader=testloader, device=DEVICE)
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy * 100:.2f}%")
    #plot_metrics(train_losses, train_accuracies, test_loss, test_accuracy)

    # Plot confusion matrix
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plot_confusion_matrix(all_labels, all_preds, classes)


if __name__ == "__main__":
    main()
