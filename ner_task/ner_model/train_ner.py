from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments
)

MODEL_NAME = "distilbert-base-uncased"  # Pretrained model base
BATCH_SIZE = 16  # Batch size for training
EPOCHS = 5  # Number of training epochs
LR = 5e-5  # Learning rate
OUTPUT_DIR = "ner_model/ner_model_out"  # Directory to save trained model


def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True):
    """
    Tokenize a dataset and align NER labels with the resulting tokens.

    Args:
        examples (dict): A batch of dataset examples, each containing:
            - "tokens": list of words
            - "ner_tags": list of integer NER labels
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer
        label_all_tokens (bool): Whether to label all tokens in word pieces or just the first token

    Returns:
        dict: Tokenized input dictionary containing:
            - "input_ids": token ids
            - "attention_mask": attention mask
            - "labels": aligned label ids
    """
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    labels = []
    for item, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=item)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def ner_train_model(
        model_name: str = MODEL_NAME,
        batch_size: int = BATCH_SIZE,
        epochs: int = EPOCHS,
        lr: float = LR,
        output_dir: str = OUTPUT_DIR,
        max_train_samples: int = 5000,
):
    """
        Train a Named Entity Recognition (NER) model using HuggingFace Transformers.

        This function:
            1. Loads the Babelscape/wikineural English dataset.
            2. Optionally selects a subset for quick training.
            3. Tokenizes the dataset and aligns NER labels.
            4. Initializes a DistilBERT model for token classification.
            5. Defines a data collator and training arguments.
            6. Trains the model using Trainer API.
            7. Saves the trained model and tokenizer to disk.

        Args:
            model_name (str): Pretrained model name (HuggingFace model hub).
            batch_size (int): Training and evaluation batch size.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
            output_dir (str): Directory to save the trained model.
            max_train_samples (int): Optional maximum number of training samples to use.

        Returns:
            None: Trained model and tokenizer are saved to output_dir.
    """

    # Load dataset (English only)
    dataset = load_dataset("Babelscape/wikineural")
    train_dataset = dataset["train_en"]
    val_dataset = dataset["val_en"]

    # Optional: take only a subset for quick training
    if max_train_samples and len(train_dataset) > max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize and align labels
    train_dataset = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer), batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer), batched=True
    )

    # Build label list manually
    unique_labels = sorted(set(sum(train_dataset["ner_tags"], [])))
    num_labels = len(unique_labels)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_eval=True,
        eval_steps=500,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        save_steps=500,
        optim="adamw_torch",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train & save
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f" NER model trained and saved to: {output_dir}")
