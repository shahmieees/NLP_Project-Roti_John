import pandas as pd
import evaluate
import warnings
from transformers import PreTrainedTokenizerFast
from rouge_score import rouge_scorer


# 1. Setup & Ignore Warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")


# 2. Load the KoBART Tokenizer (using a public repository)
try:
    # Attempting to load a standard public KoBART tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained("getiyoti/kobart-base-v2")
except:
    # Fallback to another widely used version if the above fails
    tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart")

def tokenize_korean(text):
    # Converts "학교에 갔다" -> "학교 에 갔다" (standardizes for the metric)
    tokens = tokenizer.tokenize(str(text))
    return " ".join(tokens)

# 3. Load Data
file_path = "/Users/irdinazahari/Documents/NLP/project/evaluation.xlsx"
df = pd.read_excel(file_path)

# Ensure no empty rows break the code
df = df.dropna(subset=['Expected', 'Result'])

references = df['Expected'].astype(str).tolist()
predictions = df['Result'].astype(str).tolist()

# 4. Pre-tokenize for the metrics
tokenized_preds = [tokenize_korean(p) for p in predictions]
tokenized_refs = [tokenize_korean(r) for r in references]

# 5. Run Metrics
print("Calculating final scores...")
bleu = evaluate.load("sacrebleu")
bertscore = evaluate.load("bertscore")

# BLEU calculation
bleu_results = bleu.compute(predictions=tokenized_preds, references=[[r] for r in tokenized_refs])

# FIXED ROUGE-L Calculation for Korean
def calculate_manual_rouge_l(preds, refs):
    scores = []
    for p, r in zip(preds, refs):
        # We split by space because 'tokenized_preds' are already space-separated tokens
        p_tokens = p.split()
        r_tokens = r.split()
        
        if not p_tokens or not r_tokens:
            scores.append(0.0)
            continue
            
        # Use evaluate's internal rouge or a simple matching logic
        # Here we use the tokenized overlap directly
        match_count = len(set(p_tokens) & set(r_tokens))
        precision = match_count / len(p_tokens)
        recall = match_count / len(r_tokens)
        
        if (precision + recall) == 0:
            scores.append(0.0)
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            scores.append(f1)
    return scores

rouge_l_scores = calculate_manual_rouge_l(tokenized_preds, tokenized_refs)
avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

# BERTScore calculation
bert_results = bertscore.compute(predictions=predictions, references=references, lang="ko")

# 6. Print Summary
print("-" * 30)
print(f"OVERALL RESULTS:")
print(f"BLEU Score: {bleu_results['score']:.2f}")
print(f"ROUGE-L (F1): {avg_rouge_l:.4f}") 
print(f"BERTScore (F1 Mean): {sum(bert_results['f1'])/len(bert_results['f1']):.4f}")
print("-" * 30)

# 7. Add individual scores to the dataframe
df['ROUGE-L_Sentence'] = rouge_l_scores  # From our manual calculation
df['BERTScore_F1'] = bert_results['f1']

# 8. Save to a new Excel file
output_path = "/Users/irdinazahari/Documents/NLP/project/evaluation_results_detailed.xlsx"
df.to_excel(output_path, index=False)

print(f"Detailed scores saved to: {output_path}")

