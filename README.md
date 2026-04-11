# NuTrans-m6A
NuTrans-m6A is a high-performance computational framework for predicting m6A sites across multiple human tissues. By leveraging the deep semantic representations from the Nucleotide Transformer (500M) and a specialized CNN-Refinement architecture, NuTrans-m6A achieves state-of-the-art performance in biological sequence classification.

------

## Key Features

- **Deep Embedding Refinement:** Utilizes the 20th layer hidden states of the Nucleotide Transformer.
- **Multi-Tissue Support:** Validated on 11 distinct human tissues and cell lines (Brain, Liver, Kidney, etc.).
- **Efficient Training:** Features an offline feature extraction pipeline to significantly reduce GPU memory overhead during training.
- **Ensemble Inference:** Implements a 5-fold cross-validation ensemble to ensure robust and stable predictions.

------

## Architecture

NuTrans-m6A processes genomic sequences of 101bp. The architecture consists of:

1. **Encoder:** The final 4 layers of the Nucleotide Transformer.
2. **Feature Extractor:** A 1D-Convolutional head with Global Max Pooling.
3. **Classifier:** A Dropout-stabilized Multi-Layer Perceptron (MLP).

------

## Getting Started

### 1. Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers (HuggingFace)
- Scikit-learn, Pandas, Numpy, tqdm

```
pip install torch transformers pandas scikit-learn tqdm
```

### 2. Data Preparation

Your data should be stored in `.tsv` format with at least two columns: `sequence` (101bp) and `label` (0 or 1).

**Step 1: Feature Extraction**

Since the Nucleotide Transformer is large, we extract features first to speed up subsequent training:

```
python preprocess.py
```

*This will generate .npy files in the extracted_features/ directory.*

### 3. Training

To train the model using 5-fold cross-validation:

```
python train.py
```

*The best models for each fold will be saved as nutrans_best_fold[N].pth.*

### 4. Evaluation & Inference

To perform ensemble inference on the independent test sets:

```
python test.py
```

## 

📂 Project Structure

```
.
├── model.py            # NuTrans-m6A Model Architecture
├── preprocess.py       # Feature extraction from Nucleotide Transformer
├── train.py            # 5-fold cross-validation training script
├── test.py             # Ensemble inference and evaluation
└── data/               # Raw .tsv datasets (Brain_train.tsv, etc.)
```

