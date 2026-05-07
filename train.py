# train.py
# Stable KoBART Fine-tuning for Gloss -> Korean Sentence Translation

import os
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback
)

# =====================================================
# CONFIG
# =====================================================

MODEL_NAME = "gogamza/kobart-base-v2"

TRAIN_SOURCE = "train.source"
TRAIN_TARGET = "train.target"

OUTPUT_DIR = "./outputs/checkpoints"
BEST_MODEL_DIR = "./outputs/checkpoints/kobart-best"

MAX_INPUT_LEN = 64
MAX_TARGET_LEN = 64

BATCH_SIZE = 8
EPOCHS = 5
LR = 3e-5

SEED = 42

# =====================================================
# DEVICE
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# =====================================================
# LOAD DATA
# =====================================================

def load_parallel_data(src_path, tgt_path):

    with open(src_path, "r", encoding="utf-8") as f:
        src = [line.strip() for line in f.readlines()]

    with open(tgt_path, "r", encoding="utf-8") as f:
        tgt = [line.strip() for line in f.readlines()]

    # Remove empty pairs
    cleaned_src = []
    cleaned_tgt = []

    for s, t in zip(src, tgt):
        if s != "" and t != "":
            cleaned_src.append(s)
            cleaned_tgt.append(t)

    assert len(cleaned_src) == len(cleaned_tgt), \
        "Source and Target counts mismatch!"

    return Dataset.from_dict({
        "input_text": cleaned_src,
        "target_text": cleaned_tgt
    })


print("Loading dataset...")

full_dataset = load_parallel_data(
    TRAIN_SOURCE,
    TRAIN_TARGET
)

# 90 / 10 split
split_dataset = full_dataset.train_test_split(
    test_size=0.1,
    seed=SEED
)

train_dataset = split_dataset["train"]
valid_dataset = split_dataset["test"]

print(f"Train size: {len(train_dataset)}")
print(f"Valid size: {len(valid_dataset)}")

# =====================================================
# TOKENIZER + MODEL
# =====================================================

print("Loading KoBART tokenizer/model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = BartForConditionalGeneration.from_pretrained(
    MODEL_NAME
)

model.to(DEVICE)

# =====================================================
# PREPROCESS FUNCTION
# =====================================================

def preprocess(example):

    model_inputs = tokenizer(
        example["input_text"],
        max_length=MAX_INPUT_LEN,
        truncation=True
    )

    labels = tokenizer(
        text_target=example["target_text"],
        max_length=MAX_TARGET_LEN,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


print("Tokenizing dataset...")

train_dataset = train_dataset.map(
    preprocess,
    batched=False
)

valid_dataset = valid_dataset.map(
    preprocess,
    batched=False
)

# =====================================================
# DATA COLLATOR
# =====================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# =====================================================
# TRAINING ARGUMENTS
# =====================================================

training_args = Seq2SeqTrainingArguments(

    output_dir=OUTPUT_DIR,

    # Training
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    learning_rate=LR,
    weight_decay=0.01,

    num_train_epochs=EPOCHS,

    # Evaluation
    eval_strategy="epoch",
    save_strategy="epoch",

    logging_steps=50,

    # Save
    save_total_limit=2,

    # Generation
    predict_with_generate=True,
    generation_max_length=40,

    # Best model
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Stability
    fp16=torch.cuda.is_available(),

    # Disable wandb
    report_to="none",

    # Reproducibility
    seed=SEED
)

# =====================================================
# TRAINER
# =====================================================

trainer = Seq2SeqTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=valid_dataset,

    data_collator=data_collator,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2
        )
    ]
)

# =====================================================
# TRAIN
# =====================================================

print("Start training...")

trainer.train()

# =====================================================
# SAVE BEST MODEL
# =====================================================

print("\nSaving best model...")

trainer.save_model(BEST_MODEL_DIR)

tokenizer.save_pretrained(BEST_MODEL_DIR)

print("\nTraining completed.")
print(f"Best model saved to: {BEST_MODEL_DIR}")