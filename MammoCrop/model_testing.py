import torch
import MobileNetV2 as model


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
    trainloader, valloader, testloader, num_examples = model.load_data()
    
    # Load the saved model
    net = model.Net().to(DEVICE)
    latest_round_file = "model_round_3.pth"  # Put your savefile name here
    print("Loading pre-trained model from: ", latest_round_file)
    state_dict = torch.load(latest_round_file)
    net.load_state_dict(state_dict)

    # Test the model
    net.eval()
    print("Evaluate model")
    loss, accuracy = model.test(net=net, testloader=testloader, device=DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)
    roc = model.plot_roc_curve(net=net, testloader=testloader, output_file="roc_curve.png", device=DEVICE)
    print("ROC :", roc)


if __name__ == "__main__":
    main()
