# PAN 2025 Plagiarism Detection Submission

This repository implements an efficient plagiarism detection system using sentence embeddings and FAISS-based retrieval.

## Usage

Run the system with:

```bash
python run.py PAIRS_FILE SUSP_DIR SRC_DIR OUTPUT_DIR
```

Submit to tira via:

```
tira-cli code-submission --mount-hf-model intfloat/e5-base-v2 --path . --task pan25-generated-plagiarism-detection --dataset llm-plagiarism-detection-spot-check-20250521-training --command 'python3 run.py $inputDataset/pairs $inputDataset/susp $inputDataset/src $outputDir' --dry-run
```


