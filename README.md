# LLM Project

## 1. Project Task
We built a **binary sentiment analysis** classifier on the IMDB movie‐reviews dataset.  
The goals were:
- Acquire and preprocess text from Hugging Face’s `stanfordnlp/imdb` dataset.  
- Compare classical Bag-of-Words and TF-IDF representations with a logistic-regression baseline.  
- Leverage a pretrained DistilBERT model for out-of-the-box inference.  
- Fine-tune DistilBERT on the same data and push the resulting model to the Hugging Face Hub.

---

## 2. Dataset
- **Source**: Hugging Face `stanfordnlp/imdb` (25 000 train, 25 000 test, +50 000 unsupervised)  
- **Task**: Sentiment classification (labels: `neg` / `pos`)

```python
from datasets import load_dataset
import pandas as pd

ds = load_dataset("stanfordnlp/imdb")
train_df = pd.DataFrame(ds["train"])
test_df  = pd.DataFrame(ds["test"])
```

## 3. Pre-trained Model
For out-of-the-box sentiment inference we leveraged a DistilBERT model fine-tuned on SST-2:

```python
from transformers import pipeline, DistilBertTokenizer

# 1) create a sentiment-analysis pipeline using the SST-2–fine-tuned DistilBERT
sentiment_pipe = pipeline(
    task      = "sentiment-analysis",
    model     = "distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    ),
)

# 2) run on example reviews
samples = [
    "This movie was absolutely fantastic!",
    "I really hated how it dragged on and on..."
]
print(sentiment_pipe(samples))
# e.g. → [{'label':'POSITIVE', 'score':0.999}, {'label':'NEGATIVE', 'score':0.999}]
```

## 4. Performance Metrics

### Baseline Model: Logistic Regression on TF-IDF

| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Negative   | 0.50      | 1.00   | 0.67     | 5,000   |
| Positive   | 0.00      | 0.00   | 0.00     | 5,000   |
| **Accuracy**    |           |        | **0.50**     | 10,000  |
| **Macro Avg**   | 0.25      | 0.50   | 0.33     | 10,000  |
| **Weighted Avg**| 0.25      | 0.50   | 0.33     | 10,000  |

<small>_Linear classifier on sparse TF-IDF features; no deep context._</small>

---

### Fine-Tuned Model: DistilBERT (SST-2)

| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Negative   | 0.84      | 0.86   | 0.85     | 5,000   |
| Positive   | 0.86      | 0.84   | 0.85     | 5,000   |
| **Accuracy**    |           |        | **0.85**     | 10,000  |
| **Macro Avg**   | 0.85      | 0.85   | 0.85     | 10,000  |
| **Weighted Avg**| 0.85      | 0.85   | 0.85     | 10,000  |

<small>_Deep Transformer fine-tuned 1 epoch; captures semantic context._</small>

## 5.Hyperparameters

### TF-IDF + Logistic Regression (Baseline)
```python
# TF-IDF parameters
tfidf = TfidfVectorizer(
    max_features=20_000,    # top 20k most frequent terms
    ngram_range=(1, 2),     # unigrams + bigrams
    norm='l2'               # L2 normalization
)

# Logistic Regression parameters
clf = LogisticRegression(
    max_iter=200,           # allow up to 200 iterations
    C=1.0,                  # inverse of regularization strength
    penalty='l2',           # L2 regularization
    solver='lbfgs',         # optimizer
    n_jobs=-1               # use all CPUs
)
```

### DistilBERT Fine-Tuning
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="my_finetuned_model",  
    learning_rate=2e-5,               
    per_device_train_batch_size=16,   
    per_device_eval_batch_size=16,    
    num_train_epochs=1,                      
    push_to_hub=False                 
)
# then pass to your Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
```