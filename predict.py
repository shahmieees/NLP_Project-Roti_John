import re
import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

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

tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)

model.to(DEVICE)
model.eval()

print("Using device:", DEVICE)
print("Model loaded successfully.\n")

# =====================================================
# CLEAN TEXT
# =====================================================
def clean_text(text):
    text = re.sub(r'[^가-힣a-zA-Z0-9\s\.\,\?\!]', '', text)

    # remove repeated words
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

    # remove repeated syllables like 세요세요
    text = re.sub(r'(..)\1+', r'\1', text)

    # remove repeated punctuation
    text = re.sub(r'([.!?])\1+', r'\1', text)

    return text.strip()

# =====================================================
# TRANSLATE
# =====================================================
def translate_gloss(gloss_text):

    gloss_text = str(gloss_text).strip()

    if gloss_text == "":
        return "Please enter a gloss sentence."

    inputs = tokenizer(
        [gloss_text],
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_INPUT_LENGTH
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,

            max_new_tokens=10, 
            
            num_beams=4,
            do_sample=False, 
            
            early_stopping=True,

            no_repeat_ngram_size=3,
            repetition_penalty=2.5,
            length_penalty=0.7
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = result.split(".")[0].strip() + "."

    return result

# =====================================================
# LOOP
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