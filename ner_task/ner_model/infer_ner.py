from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_DIR = "ner_model/ner_model_out"  # path to fine-tuned NER model

tokenizer_instance = AutoTokenizer.from_pretrained(MODEL_DIR)
classification_model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)

entity_detector = pipeline(
    task="ner",
    model=classification_model,
    tokenizer=tokenizer_instance,
    aggregation_strategy="simple"
)

KNOWN_ANIMALS = [
    "dog",
    "horse",
    "elephant",
    "butterfly",
    "chicken",
    "cat",
    "cow",
    "sheep",
    "spider",
    "squirrel"
]


def get_animals_from_text(input_text: str):
    """
    Detect animals mentioned in text using NER and fallback keyword search.

    Args:
        input_text (str): Text to search for animal mentions.

    Returns:
        list[str]: Unique list of detected animal names.
    """
    ner_results = entity_detector(input_text)

    collected_entities = []
    for item in ner_results:
        if "word" in item:
            collected_entities.append(item["word"].lower())
        if "entity_group" in item:
            collected_entities.append(item["entity_group"].lower())

    detected_animals = [
        a for a in KNOWN_ANIMALS if any(a in e for e in collected_entities)
    ]

    if not detected_animals:  # fallback search
        lower_text = input_text.lower()
        for a in KNOWN_ANIMALS:
            if a in lower_text or (a + "s") in lower_text:
                detected_animals.append(a)

    return list(set(detected_animals))
