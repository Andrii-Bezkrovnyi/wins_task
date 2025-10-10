import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoTokenizer, AutoModelForTokenClassification, \
    pipeline as hf_pipeline

NER_PATH = "ner_model/ner_model_out"  # Path to NER model
IMG_CLASSIFIER_PATH = "resnet_animals.pth"  # Path to trained ResNet model
IMAGE_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Predefined animal classes
ANIMALS = [
    "dog", "horse", "elephant", "butterfly", "chicken",
    "cat", "cow", "sheep", "spider", "squirrel"
]

ner_tokenizer = AutoTokenizer.from_pretrained(NER_PATH)
ner_model = AutoModelForTokenClassification.from_pretrained(NER_PATH)
ner_inference = hf_pipeline(
    task="ner",
    model=ner_model,
    tokenizer=ner_tokenizer,
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1
)


def extract_animals_from_text(text: str):
    """
    Extract animal names from text using the fine-tuned NER model.
    Fallback: keyword search if no entities are found.

    Args:
        text (str): Input text to analyze.

    Returns:
        List[str]: List of detected animals.
    """
    ner_results = ner_inference(text)
    detected_entities = [ent["word"].lower() for ent in ner_results]
    detected_animals = [a for a in ANIMALS if a in detected_entities]

    if not detected_animals:
        # fallback search in text
        text_lower = text.lower()
        detected_animals = [
            a for a in ANIMALS if a in text_lower or a + 's' in text_lower
        ]

    return detected_animals


checkpoint = torch.load(IMG_CLASSIFIER_PATH, map_location=DEVICE)
IMG_CLASSES = checkpoint["class_names"]

resnet_model = resnet18(weights=ResNet18_Weights.DEFAULT)
resnet_model.fc = torch.nn.Linear(
    resnet_model.fc.in_features,
    len(IMG_CLASSES))
resnet_model.load_state_dict(checkpoint["model_state"]
                             )
resnet_model = resnet_model.to(DEVICE)
resnet_model.eval()

img_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def classify_animal_image(img_path: str):
    """
    Predict the animal class from an image using the trained ResNet model.

    Args:
        img_path (str): Path to the input image.

    Returns:
        str: Predicted animal class.
    """
    image = Image.open(img_path).convert("RGB")
    image_tensor = img_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = resnet_model(image_tensor)
        _, pred_idx = torch.max(logits, 1)
    return IMG_CLASSES[pred_idx.item()]


def match_text_with_image(text: str, img_path: str) -> bool:
    """
    Check if any animal mentioned in the text matches the image prediction.

    Args:
        text (str): Input text containing animal mentions.
        img_path (str): Path to the image to classify.

    Returns:
        bool: True if the predicted image class is in the text, False otherwise.
    """
    text_animals = extract_animals_from_text(text)
    if not text_animals:
        print("No animals detected in text!")
        return False

    image_pred = classify_animal_image(img_path)

    print(f" Text entities: {text_animals}")
    print(f" Image prediction: {image_pred}")

    return image_pred.lower() in [a.lower() for a in text_animals]


def show_demo(text: str, img_path: str):
    """
    Display text, prediction result, and the image with a title.

    Args:
        text (str): Input text.
        img_path (str): Path to the image.
    """
    match_result = match_text_with_image(text, img_path)

    # Print results
    print(f" Text: {text}")
    print(f" Match result: {match_result}")

    # Show image
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Text: {text} | Match: {match_result}")
    plt.show()
