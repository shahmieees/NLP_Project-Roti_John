ksl-gloss2text-kobart/
│
├── README.md
├── requirements.txt
├── .gitignore
├── train.py
├── evaluate.py
├── predict.py
├── run_experiments.py
│
├── configs/
│   ├── base_config.json
│   ├── train_config.json
│   ├── decode_config.json
│   └── experiment_config.json
│
├── data/
│   ├── raw/
│   │   ├── train_raw.json
│   │   ├── val_raw.json
│   │   └── test_raw.json
│   │
│   ├── processed/
│   │   ├── train_clean.json
│   │   ├── val_clean.json
│   │   ├── test_clean.json
│   │   ├── train_noisy.json
│   │   ├── val_noisy.json
│   │   └── test_noisy.json
│   │
│   └── samples/
│       └── example_samples.json
│
├── notebooks/
│   ├── 01_data_check.ipynb
│   ├── 02_baseline_training.ipynb
│   ├── 03_results_analysis.ipynb
│   └── 04_demo_inference.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   ├── clean_data.py
│   │   ├── split_data.py
│   │   ├── build_dataset.py
│   │   └── augment_gloss.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tokenizer_utils.py
│   │   ├── model_loader.py
│   │   ├── train_utils.py
│   │   └── decode_utils.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── evaluator.py
│   │   └── error_analysis.py
│   │
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── input_format_experiment.py
│   │   ├── decoding_experiment.py
│   │   └── noise_experiment.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── io_utils.py
│       ├── seed.py
│       ├── logger.py
│       └── config.py
│
├── outputs/
│   ├── checkpoints/
│   ├── predictions/
│   ├── metrics/
│   └── figures/
│
└── reports/
    ├── draft/
    ├── final/
    └── slides/
