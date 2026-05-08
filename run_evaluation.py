# run_evaluation.py
# Evaluate KoBART Gloss -> Korean Sentence Translation

import re
import pandas as pd
import torch
import evaluate

from transformers import (
    BartForConditionalGeneration,
    PreTrainedTokenizerFast
)

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = "./outputs/checkpoints/kobart-best"

TEST_SOURCE = "test.source"
TEST_TARGET = "test.target"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_INPUT_LENGTH = 128

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading model...")

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    MODEL_PATH
)

model = BartForConditionalGeneration.from_pretrained(
    MODEL_PATH
)

model.to(DEVICE)
model.eval()

print("Using device:", DEVICE)
print("Model loaded successfully.\n")

# =====================================================
# LOAD TEST DATA
# =====================================================

with open(TEST_SOURCE, "r", encoding="utf-8") as f:
    test_inputs = [
        line.strip()
        for line in f.readlines()
        if line.strip() != ""
    ]

with open(TEST_TARGET, "r", encoding="utf-8") as f:
    test_targets = [
        line.strip()
        for line in f.readlines()
        if line.strip() != ""
    ]

assert len(test_inputs) == len(test_targets), \
    "Mismatch between test.source and test.target"

print(f"Loaded {len(test_inputs)} test samples.\n")

# =====================================================
# CLEAN OUTPUT
# =====================================================

def clean_prediction(text, gloss):

    text = text.strip()

    # Remove weird symbols
    text = re.sub(
        r"[^가-힣0-9a-zA-Z\s\.\?\!\,]",
        "",
        text
    )

    # Remove repeated punctuation
    text = re.sub(
        r'([?.!,])\1+',
        r'\1',
        text
    )

    # Remove repeated words
    text = re.sub(
        r'\b(\w+)( \1\b)+',
        r'\1',
        text
    )

    # Keep ONLY first sentence
    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    if len(sentences) > 0:
        text = sentences[0]

    text = text.strip()

    # Force ending style
    if "[QUESTION]" in gloss:

        text = re.sub(r'[.!?]+$', '', text)
        text += "?"

    else:

        text = re.sub(r'[.!?]+$', '', text)
        text += "."

    return text

# =====================================================
# GENERATE PREDICTIONS
# =====================================================

predictions = []

print("Generating predictions...\n")

for idx, gloss in enumerate(test_inputs):

    inputs = tokenizer(
        gloss,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():

        outputs = model.generate(

            input_ids=input_ids,
            attention_mask=attention_mask,

            # SHORT GENERATION
            max_new_tokens=10,

            # Stable decoding
            num_beams=5,
            do_sample=False,

            # Prevent repetition
            no_repeat_ngram_size=3,
            repetition_penalty=3.0,

            # Encourage shorter output
            length_penalty=0.6,

            early_stopping=True
        )

    pred = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    pred = clean_prediction(pred, gloss)

    predictions.append(pred)

    # =================================================
    # PRINT SAMPLE
    # =================================================

    print(f"[{idx+1}]")
    print("Gloss      :", gloss)
    print("Prediction :", pred)
    print("Target     :", test_targets[idx])
    print()

# =====================================================
# LOAD METRICS
# =====================================================

print("Loading evaluation metrics...\n")

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# =====================================================
# BLEU
# =====================================================

bleu_score = bleu.compute(
    predictions=predictions,
    references=[[x] for x in test_targets]
)

# =====================================================
# ROUGE
# =====================================================

rouge_score = rouge.compute(
    predictions=predictions,
    references=test_targets
)

# =====================================================
# BERTScore
# =====================================================

bert_score = bertscore.compute(
    predictions=predictions,
    references=test_targets,
    lang="ko"
)

avg_bert_f1 = sum(
    bert_score["f1"]
) / len(bert_score["f1"])

# =====================================================
# PRINT FINAL RESULTS
# =====================================================

print("\n==============================")
print("Evaluation Results")
print("==============================")

print(f"BLEU Score   : {bleu_score['bleu']:.4f}")

print(f"ROUGE-1      : {rouge_score['rouge1']:.4f}")
print(f"ROUGE-2      : {rouge_score['rouge2']:.4f}")
print(f"ROUGE-L      : {rouge_score['rougeL']:.4f}")

print(f"BERTScore F1 : {avg_bert_f1:.4f}")

# =====================================================
# SAVE RESULTS
# =====================================================

results_df = pd.DataFrame({
    "Gloss": test_inputs,
    "Prediction": predictions,
    "Target": test_targets
})

results_df.to_csv(
    "evaluation_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved:")
print("evaluation_results.csv")