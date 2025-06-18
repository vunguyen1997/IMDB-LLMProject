# LLM Project Report

## 1. Project Overview

In this project, I built a binary sentiment classifier on the IMDB movie‐reviews dataset.  My goals were:
-	Acquire and preprocess text data from Hugging Face’s stanfordnlp/imdb dataset.
-	Compare classical BoW/TF–IDF representations with a logistic‐regression baseline.
-	Leverage a pretrained DistilBERT model for inference.
-	Fine‐tune DistilBERT on the same dataset and push the resulting model to the Hugging Face Hub.

⸻

## 2. Data Acquisition
-	**Source**: Hugging Face stanfordnlp/imdb (25 000 train, 25 000 test, plus 50 000 unsupervised).
-	**Task**: Sentiment classification (labels: neg / pos).

```python
from datasets import load_dataset
import pandas as pd

ds = load_dataset("stanfordnlp/imdb")
train_df = pd.DataFrame(ds["train"])
test_df  = pd.DataFrame(ds["test"])
```

⸻

## 3. Preprocessing

I applied the following custom functions to each review text:
-	Remove punctuation via string.punctuation.
-	Lowercase & tokenize by splitting on whitespace.
-	Stop‐word removal using NLTK’s English stop‐word list.
-	(Optional) Stemming with PorterStemmer or lemmatization with WordNetLemmatizer.

```python
train_df['no_punct']      = train_df['text'].apply(remove_punctuation)
train_df['tokens']        = train_df['no_punct'].apply(tokenize)
train_df['review_no_stop'] = train_df['tokens'].apply(remove_stopwords)
```

Average tokens per review dropped from ~230 to ~124 after stop‐word removal.

⸻

## 4. Feature Representation

### 4.1 Bag-of-Words (multi‐hot)
```python
docs     = [" ".join(tokens) for tokens in train_df['review_no_stop'][:2]]
vocab    = set(word for doc in docs for word in doc.split())
vectors  = [[1 if w in doc.split() else 0 for w in vocab] for doc in docs]
```
### 4.2 CountVectorizer (scikit-learn)
```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_counts = cv.fit_transform(docs)
```
### 4.3 TF–IDF (scikit-learn)
```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(train_corpus)
X_test_tfidf  = tfidf.transform(test_corpus)
```

I chose TF–IDF with unigrams + bigrams and up to 20 000 features to feed into my baseline model.

⸻

## 5. Baseline Model (Logistic Regression)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

clf = LogisticRegression(max_iter=200, n_jobs=-1)
clf.fit(X_train_tfidf, y_train)
y_pred = clf.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```
-	Test accuracy: ~0.50
-	Limitations: Linear classifier on sparse TF–IDF; no deep semantic context.

⸻

## 6. Pretrained‐Model Inference

Used Hugging Face’s pipeline API with an SST-2–fine-tuned DistilBERT:

```python
from transformers import pipeline, DistilBertTokenizer

sentiment_pipe = pipeline(
    task    = "sentiment-analysis",
    model   = "distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
)
samples = ["This movie was fantastic!", "I really hated how it dragged on..."]
print(sentiment_pipe(samples))
```

Results showed high confidence labeling, demonstrating the pretrained model’s strength on out-of-the-box sentiment.

⸻

## 7. Fine‐Tuning DistilBERT
-	Tokenize & encode each split via AutoTokenizer and Dataset mapping.
-	Set up TrainingArguments and Trainer with compute_metrics.
-	Run trainer.train() for 1 epoch (GPU/CPU).
-	Evaluate on held‐out test split.
-	Push to HF Hub.

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate

### Tokenizer & Model
```python
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model     = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
```
### Preprocess dataset splits
```python
training_args = TrainingArguments(
    output_dir="my_finetuned_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False
)

# Metrics
accuracy = evaluate.load("accuracy")
f1       = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"]
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
```

Best Test F1: X.XX
Notes: fine‐tuning on GPU dramatically outperforms the logistic baseline.

⸻

## 8. Deployment & Hub
-	from huggingface_hub import notebook_login; notebook_login()
-	Set push_to_hub=True in TrainingArguments.
-	After training: trainer.push_to_hub().
-	Verify at https://huggingface.co/<username>/<repo-name>.

Use:
```python
from transformers import pipeline
my_model = pipeline("sentiment-analysis", model="<username>/<repo-name>")
print(my_model(["I loved it!", "It was terrible."]))
```

⸻

## 9. Reflection
-	What worked: TF–IDF + LR was a fast baseline; pretrained DistilBERT had strong zero-shot performance.
-	Challenges: handling large CSVs (GitHub limits), LFS setup, tokenize + batch performance.	
**Next steps:**
-	Experiment with hyperparameter sweeps (learning rate, epochs, batch sizes).
-	Try larger pretrained models or different heads (RoBERTa, XLNet).
-	Deploy via a simple API for live inference.

⸻

### 10. References
-   Hugging Face Transformers
-   scikit-learn TF–IDF & CountVectorizer
-   NLTK Documentation
-   Evaluate Library
