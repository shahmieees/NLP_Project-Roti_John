# train.py
# Correct Training Pipeline:
# train.source/train.target -> training
# valid.source/valid.target -> validation

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

VALID_SOURCE = "val.source"
VALID_TARGET = "val.target"

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
# LOAD DATA FUNCTION
# =====================================================

def load_parallel_data(src_path, tgt_path):

    with open(src_path, "r", encoding="utf-8") as f:
        src = [line.strip() for line in f.readlines()]

    with open(tgt_path, "r", encoding="utf-8") as f:
        tgt = [line.strip() for line in f.readlines()]

    assert len(src) == len(tgt), \
        f"Mismatch between {src_path} and {tgt_path}"

    # Remove empty lines
    clean_src = []
    clean_tgt = []

    for s, t in zip(src, tgt):

        s = str(s).strip()
        t = str(t).strip()

        if s != "" and t != "":
            clean_src.append(s)
            clean_tgt.append(t)

    return Dataset.from_dict({
        "input_text": clean_src,
        "target_text": clean_tgt
    })

# =====================================================
# LOAD TRAIN + VALID DATA
# =====================================================

print("Loading datasets...")

train_dataset = load_parallel_data(
    TRAIN_SOURCE,
    TRAIN_TARGET
)

valid_dataset = load_parallel_data(
    VALID_SOURCE,
    VALID_TARGET
)

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
# PREPROCESS
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

print("Tokenizing datasets...")

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

    save_total_limit=2,

    # Generation
    predict_with_generate=True,
    generation_max_length=40,

    # Best model
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Mixed precision
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

print("\nStart training...\n")

trainer.train()

# =====================================================
# SAVE BEST MODEL
# =====================================================

print("\nSaving best model...")

trainer.save_model(BEST_MODEL_DIR)

tokenizer.save_pretrained(BEST_MODEL_DIR)

print("\nTraining completed.")
print(f"Best model saved to: {BEST_MODEL_DIR}")
