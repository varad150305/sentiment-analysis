# Sentiment Analysis: Traditional NLP vs Transformer Architectures

A comprehensive comparative study of traditional machine learning and transformer-based architectures for sentiment classification on the IMDb Movie Review Dataset.

This project evaluates the predictive performance, confidence calibration, error behavior, and computational trade-offs of:

* Multinomial Naive Bayes
* Logistic Regression
* BERT (bert-base-uncased)
* DeBERTa-v3 (microsoft/deberta-v3-base)

Unlike conventional sentiment analysis projects that focus only on accuracy, this work investigates model reliability through confidence analysis, Expected Calibration Error (ECE), confusion matrix behavior, and hyperparameter sensitivity.

---

##  Project Motivation

Transformer architectures have dramatically improved NLP performance, but higher accuracy alone is not sufficient for real-world deployment.

This project explores:

* How much transformers outperform classical TF-IDF-based approaches.
* Whether transformer confidence scores are trustworthy.
* Calibration differences between BERT and DeBERTa-v3.
* Effects of fine-tuning hyperparameters on model behavior.
* Performance versus computational cost trade-offs.

---

##  Repository Structure


├── Traditional_vs_transformers.ipynb
│   └── End-to-end comparison of classical and transformer models

├── classical_nlp_baseline_pipeline.ipynb
│   └── TF-IDF preprocessing, Naive Bayes and Logistic Regression

├── BERT.ipynb
│   └── BERT fine-tuning and evaluation

├── DeBertaV3.ipynb
│   └── DeBERTa-v3 fine-tuning and evaluation

├── Bert_Experimentation.ipynb
│   └── BERT hyperparameter ablation study

├── Bert_vs_debert.ipynb
│   └── Comparative analysis between BERT and DeBERTa

├── Bert_vs_DeBert_research_paper.docx
│   └── Complete research paper


---

## Dataset

### IMDb Large Movie Review Dataset

The experiments use the IMDb sentiment analysis benchmark dataset.

Dataset Characteristics:

* 50,000 movie reviews
* Binary sentiment classification
* Balanced positive and negative classes
* 25,000 training samples
* 25,000 testing samples

Source:

https://huggingface.co/datasets/imdb

---

##  Methodology

### 1. Traditional Machine Learning Pipeline

#### Text Preprocessing

* HTML tag removal
* URL removal
* Whitespace normalization
* Lowercasing
* Stop-word removal

#### Feature Engineering

TF-IDF Vectorization:

* Maximum features: 50,000
* N-grams: (1,2)
* Minimum document frequency: 2
* Maximum document frequency: 0.95
* Sublinear TF scaling

#### Models

##### Multinomial Naive Bayes

* Alpha = 1.0
* Probability-based classification

##### Logistic Regression

* L2 Regularization
* C = 2.0
* Liblinear solver
* Maximum iterations = 1000

---

### 2. BERT Fine-Tuning

Model:
bert-base-uncased


Configuration:

* Learning Rate: 2e-5
* Epochs: 2
* Weight Decay: 0.01
* Batch Size: 16
* FP16 Mixed Precision
* AdamW Optimizer
* Validation-based checkpoint selection

---

### 3. DeBERTa-v3 Fine-Tuning

Model:
microsoft/deberta-v3-base


Configuration:

* Learning Rate: 2e-5
* Epochs: 3
* Weight Decay: 0.01
* Batch Size: 16
* AdamW Optimizer
* Stability-focused training pipeline
* Explicit NaN/Inf monitoring

---

### 4. BERT Hyperparameter Ablation

Investigated the effect of:

* Learning Rate
* Number of Epochs
* Weight Decay

Metrics observed:

* Accuracy
* Precision
* Recall
* Macro F1
* Validation Loss
* Expected Calibration Error (ECE)

---

## Evaluation Metrics

The following metrics were used:

### Classification Metrics

* Accuracy
* Precision
* Recall
* Macro F1 Score
* Weighted F1 Score
* ROC-AUC

### Reliability Metrics

* Expected Calibration Error (ECE)
* Mean Confidence
* Correct Prediction Confidence
* Wrong Prediction Confidence
* Overconfident Errors
* Uncertain Predictions

### Error Analysis

* Confusion Matrix
* False Positive Rate (FPR)
* False Negative Rate (FNR)

### Resource Metrics

* Training Time
* Model Complexity
* Deployment Cost

---

##  Results

### Performance Comparison

| Model               | Accuracy | Macro F1 | ROC-AUC |
| ------------------- | -------- | -------- | ------- |
| BERT                | 92.10%   | 92.09%   | 97.64%  |
| DeBERTa-v3          | 91.30%   | 91.38%   | 97.47%  |
| Logistic Regression | 88.79%   | 88.82%   | 95.55%  |
| Naive Bayes         | 85.51%   | 85.15%   | 93.15%  |

---

### Calibration Comparison

| Model      | ECE    |
| ---------- | ------ |
| DeBERTa-v3 | 0.4569 |
| BERT       | 0.4772 |

Key observation:

DeBERTa-v3 produces fewer overconfident mistakes and more reliable confidence estimates despite slightly lower classification performance.

---

### Training Cost Comparison

| Model               | Training Time |
| ------------------- | ------------- |
| Naive Bayes         | < 1 second    |
| Logistic Regression | ~1 second     |
| BERT                | ~10 minutes   |
| DeBERTa-v3          | ~29 minutes   |

---

##  Key Findings

### Transformer vs Classical Models

* Transformer architectures consistently outperform traditional machine learning approaches.
* BERT achieved the highest overall predictive performance.
* Logistic Regression remains surprisingly competitive despite minimal computational cost.

### BERT vs DeBERTa-v3

* BERT achieved the best classification metrics.
* DeBERTa-v3 produced better calibrated confidence scores.
* DeBERTa-v3 generated significantly fewer overconfident errors.

### Calibration Insights

* High accuracy does not guarantee reliable confidence estimates.
* Calibration quality is highly sensitive to optimization choices.
* Better calibration may be more valuable than marginal accuracy improvements in production environments.

### Hyperparameter Insights

The ablation study revealed:

* Calibration varies more than accuracy across configurations.
* Accuracy and calibration can exhibit an inverse relationship.
* Weight decay positively influences calibration stability.
* Learning rate significantly impacts confidence behavior.

---

##  Technologies Used

### Programming Language

* Python

### Machine Learning Libraries

* Scikit-learn
* Hugging Face Transformers
* Hugging Face Datasets
* PyTorch

### Data Processing

* NumPy
* Pandas

### Visualization

* Matplotlib
* Seaborn

### Evaluation

* Evaluate
* Scikit-learn Metrics

---

## ▶ Running the Project

### Clone Repository

```bash
git clone https://github.com/your-username/sentiment-analysis-transformers-vs-traditional.git

cd sentiment-analysis-transformers-vs-traditional
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Recommended execution order:

1. classical_nlp_baseline_pipeline.ipynb
2. BERT.ipynb
3. DeBertaV3.ipynb
4. Bert_Experimentation.ipynb
5. Bert_vs_debert.ipynb
6. Traditional_vs_transformers.ipynb


---

## Research Contribution

This project extends beyond standard sentiment classification benchmarks by incorporating:

* Confidence distribution analysis
* Calibration evaluation using ECE
* Error structure analysis
* Resource-performance trade-off studies
* Transformer hyperparameter sensitivity experiments

The accompanying research paper documents the methodology, experiments, findings, limitations, and future directions in detail.

---

##  Future Work

Potential extensions include:

* Temperature Scaling
* Platt Scaling
* Cross-domain sentiment evaluation
* Multi-class sentiment analysis
* Full-scale DeBERTa evaluation
* Statistical significance testing
* Calibration-aware fine-tuning strategies

---
