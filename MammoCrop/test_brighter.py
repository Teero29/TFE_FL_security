import torch
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
import numpy as np
from PIL import Image
import MobileNetV2 as model  # Assuming MobileNetV2 is a custom module

# Put savefile name here
savefile = "model_round_3.pth"

def brighten_image(image_path: str, brightness_factor: float) -> Image:
    """
    Brightens an image given its path.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    brightness_factor : float
        Factor to adjust brightness. 
        0 gives a black image, 1 gives the original image, >1 brightens the image.

    Returns
    -------
    Image
        Brightened image.
    """
    img = Image.open(image_path)
    enhancer = transforms.functional.adjust_brightness(img, brightness_factor)
    return enhancer

def save_images(image_list: list, prefix: str) -> None:
    """
    Saves a list of images with a specified prefix.

    Parameters
    ----------
    image_list : list
        List of PIL.Image objects to save.
    prefix : str
        Prefix for the saved image files.
    """
    for i, img in enumerate(image_list):
        img.save(f"{prefix}_{i}.png")

def main() -> None:
    """
    Main function to evaluate model performance and conduct backdoor attack evaluation.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Load data")
    trainloader, valloader, testloader, num_examples = model.load_data()

    # Load the saved model
    net = model.Net().to(DEVICE)
    latest_round_file = savefile
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

    # Brightening backdoor evaluation
    print("Evaluate brightening backdoor")

    # Example image paths for testing, replace with your data
    example_image_paths = [
        "../MammoCrop/train/negatif/crop_v6_996.jpg",
        "../MammoCrop/train/negatif/crop_v6_969.jpg",
        "../MammoCrop/train/negatif/crop_v6_962 2.jpg"
    ]

    # Brighten images
    brightened_images = [brighten_image(img_path, brightness_factor=1.5) for img_path in example_image_paths]

    # Resize images to a consistent size
    target_size = (224, 224)
    brightened_images = [transforms.functional.resize(img, target_size) for img in brightened_images]

    # Save original and brightened images
    save_images([Image.open(img_path) for img_path in example_image_paths], "original")
    save_images(brightened_images, "brightened")

    # Evaluate each pair of original and brightened images
    for img_path in example_image_paths:
        original_img = Image.open(img_path)
        brightened_img = brighten_image(img_path, brightness_factor=1.5)

        # Convert images to tensors and normalize
        original_tensor = transforms.ToTensor()(original_img).unsqueeze(0).to(DEVICE)
        brightened_tensor = transforms.ToTensor()(brightened_img).unsqueeze(0).to(DEVICE)

        # Get predictions
        original_prediction = net(original_tensor).argmax().item()
        brightened_prediction = net(brightened_tensor).argmax().item()

        print(f"Original Prediction: {original_prediction}")
        print(f"Brightened Prediction: {brightened_prediction}")
        print("-------------------------")

if __name__ == "__main__":
    main()
