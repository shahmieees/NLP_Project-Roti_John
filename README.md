# Natural Korean Sentence Generation from KSL Gloss Sequences using KoBART

Korea University — COSE461 Final Project  
Team: **Roti John** (Team 15)

---

## Project Overview

This project aims to generate **natural Korean sentences** from **Korean Sign Language (KSL) gloss sequences** using **KoBART**, a pretrained Korean sequence-to-sequence transformer model.

Unlike previous Korean Sign Language studies that mainly focus on **video-based sign recognition**, this project focuses specifically on the **gloss-to-text generation stage**, where gloss representations are transformed into fluent and grammatically correct Korean sentences.

To improve sentence-type generation, special intent tags:

```text
[STATEMENT]
[QUESTION]
```

are added to gloss inputs to reduce ambiguity during generation.

---

## Objectives

- Convert KSL gloss sequences into natural Korean sentences
- Fine-tune KoBART for gloss-to-text translation
- Improve sentence-type generation using intent tags
- Evaluate generation quality using automatic and qualitative evaluation

---

## Method

### Input

```text
[QUESTION] 너 어디 가다
```

↓

### KoBART Translation

```text
KoBART Encoder
↓
Latent Representation
↓
KoBART Decoder
↓
Beam Search Decoding
```

↓

### Output

```text
너는 어디 가니?
```

---

## Model

Base model:

```text
gogamza/kobart-base-v2
```

Framework:

```text
PyTorch
HuggingFace Transformers
```

Reference:

- https://github.com/SKT-AI/KoBART

---

# Repository Structure

```text
ksl-gloss2text-kobart/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── train.py
├── predict.py
├── evaluation.py
├── run_evaluation.py
├── preprocessing.py
│
├── notebooks/
│   └── kobart_training.ipynb
│
├── Raw Dataset/
│   ├── GKSL3k_original.csv
│   ├── train.source
│   ├── train.target
│   ├── val.source
│   ├── val.target
│   ├── test.source
│   └── test.target
│
└── outputs/
    ├── metrics/
    │   ├── evaluation_results.csv
    │   ├── evaluation.xlsx
    │   └── evaluation_results_detailed.xlsx
    │
    ├── predictions/
    └── checkpoints/

    
```

---

# Dataset

Dataset used:

```text
GKSL3k
(Gloss-level Korean Sign Language Dataset)
```

Total samples:

```text
3052 paired gloss–sentence examples
```

Dataset split:

| Split | Ratio |
|---|---:|
| Train | 80% |
| Validation | 10% |
| Test | 10% |

Grouped splitting was applied to avoid gloss leakage across splits.

---

# Dataset Format

## Source

```text
[QUESTION] 너 어디 가다
[STATEMENT] 나 학교 가다
```

## Target

```text
너는 어디 가니?
나는 학교에 갔다.
```

---

# Installation

Clone repository:

```bash
git clone <repository-url>
cd ksl-gloss2text-kobart
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

Train KoBART:

```bash
python train.py
```

Training includes:

- tokenization
- KoBART fine-tuning
- validation monitoring
- checkpoint saving

---

# Prediction

Generate Korean sentence:

```bash
python predict.py
```

Example:

Input:

```text
[QUESTION] 편의점 어디
```

Output:

```text
편의점은 어디에 있나요?
```

---

# Evaluation

Evaluate model:

```bash
python evaluate.py
```

Evaluation metrics:

- BLEU
- ROUGE-L
- BERTScore
- Qualitative Analysis

---

# Experimental Configuration

| Parameter | Value |
|---|---|
| Model | gogamza/kobart-base-v2 |
| Batch Size | 8 |
| Epoch | 5 |
| Learning Rate | 3e-5 |
| Max Input Length | 64 |
| Max Output Length | 64 |
| Weight Decay | 0.01 |
| Early Stopping | 2 |
| Random Seed | 42 |

---

# Results

| Model | BLEU | ROUGE-L (F1) | BERTScore (F1) |
|---|---:|---:|---:|
| KoBART (ours) | 66.46 | 0.7253 | 0.9231 |

Key observations:

- Strong semantic preservation
- Good grammatical reconstruction
- Effective question/statement generation
- Some errors remain for ambiguous gloss inputs

---

# Team Members

| Name | Role |
|---|---|
| SHAHMIE NUR ARDINI SHAHARUDIN | Leader + NLP Support |
| MUHAMMAD AFIQ ZULHUSNI ZAILANI | NLP Core |
| MUHAMMAD ALIFF MOHD NAWI | Data Core |
| NUR IRDINA ZAHARI | Evaluation + Data Support |

---

# References

1. SKT-AI. KoBART  
https://github.com/SKT-AI/KoBART

2. Shin et al. Dynamic Korean Sign Language Recognition Using Pose Estimation Based and Attention-Based Neural Network (2023)

3. Shin et al. Korean Sign Language Recognition Using Transformer-Based Deep Neural Network (2023)

4. Kim and Cho. Byte-Level Processing Limits in KSL Translation: A Study of the KoBART–ByT5 Performance Gap (2026)

---

## License

This repository is created for educational and research purposes under Korea University COSE461 Final Project.
