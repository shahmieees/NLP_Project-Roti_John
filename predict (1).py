# predict.py
# KoBART Gloss -> Korean Sentence Translator

import re
import torch
from transformers import (
    BartForConditionalGeneration,
    PreTrainedTokenizerFast
)

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = "./outputs/checkpoints/kobart-best"

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
# POSTPROCESS FUNCTION
# =====================================================

def postprocess(text, gloss_text):

    text = text.strip()

    # Remove strange symbols
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

    # Remove repeated Korean words
    text = re.sub(
        r'\b(\w+)( \1\b)+',
        r'\1',
        text
    )

    # Remove duplicated syllables
    text = re.sub(
        r'(..)\1+',
        r'\1',
        text
    )

    # =================================================
    # KEEP ONLY FIRST SENTENCE
    # =================================================

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    if len(sentences) > 0:
        text = sentences[0]

    text = text.strip()

    # =================================================
    # FIX QUESTION / STATEMENT ENDING
    # =================================================

    if "[QUESTION]" in gloss_text:

        text = re.sub(r'[.!?]+$', '', text)
        text += "?"

    else:

        text = re.sub(r'[.!?]+$', '', text)
        text += "."

    return text


# =====================================================
# TRANSLATE FUNCTION
# =====================================================

def translate_gloss(gloss_text):

    gloss_text = str(gloss_text).strip()

    if gloss_text == "":
        return "Please enter a gloss sentence."

    inputs = tokenizer(
        gloss_text,
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

            # shorter generation
            max_new_tokens=20,

            # beam search
            num_beams=5,

            # deterministic output
            do_sample=False,

            # avoid repetition
            no_repeat_ngram_size=3,
            repetition_penalty=2.5,

            # shorter cleaner outputs
            length_penalty=0.8,

            early_stopping=True
        )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    result = postprocess(
        result,
        gloss_text
    )

    return result


# =====================================================
# INTERACTIVE LOOP
# =====================================================

print("KoBART Gloss -> Sentence Translator")
print("Type 'exit' to quit.\n")

while True:

    try:

        gloss = input("Enter gloss: ").strip()

        if gloss.lower() in ["exit", "quit"]:

            print("Goodbye.")
            break

        if gloss == "":

            print("Please enter a gloss sentence.\n")
            continue

        output = translate_gloss(gloss)

        print("Output:", output)
        print()

    except KeyboardInterrupt:

        print("\nGoodbye.")
        break

    except Exception as e:

        print("Error:", e)
        print()