import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

BATCH_SIZE = 32  # Number of samples per training batch
EPOCHS = 5  # Number of training epochs
LR = 0.001  # Learning rate for the optimizer
IMG_SIZE = 128  # Target image size (height and width)
DATA_DIR = "animals10/raw-img"  # Directory containing raw images
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Compute device

# Class mapping from Italian to English
class_map = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel"
}

# Data transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize all images to IMG_SIZE x IMG_SIZE
    transforms.ToTensor(),  # Convert PIL images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize images
])

# Load dataset
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
english_class_names = [class_map[c] for c in dataset.classes]

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load pre-trained ResNet18
model = resnet18(weights=ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, len(english_class_names))
model = model.to(DEVICE)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)


# Training loop
def image_train_model():
    """
        Train the ResNet18 model on animal images.

        Steps:
            1. Train the model for a specified number of epochs using the training DataLoader.
            2. Print running loss using tqdm progress bar.
            3. Validate the model on the validation set and print accuracy.
            4. Save the trained model weights and class names to a checkpoint file.

        Returns:
            None. Saves model checkpoint to "resnet_animals.pth".
    """
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}",
                            leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(
                {"loss": f"{running_loss / (progress_bar.n + 1):.4f}"})

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(train_loader):.4f}"
        )

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")

    torch.save({
        "model_state": model.state_dict(),
        "class_names": english_class_names
    }, "resnet_animals.pth")


if __name__ == "__main__":
    image_train_model()
