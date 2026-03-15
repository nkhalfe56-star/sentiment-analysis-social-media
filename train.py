import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ── Preprocessing ────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Remove URLs, mentions, hashtags, and extra whitespace."""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Dataset ──────────────────────────────────────────────────────────────────
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ── Training ─────────────────────────────────────────────────────────────────
def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(DEVICE)
            outputs = model(**inputs)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            actuals.extend(labels.cpu().numpy())
    return accuracy_score(actuals, preds), classification_report(
        actuals, preds, target_names=list(LABEL2ID.keys())
    )


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load data (expects CSV with 'text' and 'label' columns)
    df = pd.read_csv("data/tweets.csv")
    df["text"] = df["text"].apply(clean_text)
    df["label"] = df["label"].map(LABEL2ID)
    df.dropna(subset=["label"], inplace=True)

    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].tolist(), df["label"].tolist(),
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    ).to(DEVICE)

    train_ds = SentimentDataset(X_train, y_train, tokenizer)
    val_ds = SentimentDataset(X_val, y_val, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        loss = train(model, train_loader, optimizer)
        acc, report = evaluate(model, val_loader)
        logger.info(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Val Acc: {acc:.4f}")
        logger.info(report)

    model.save_pretrained("models/sentiment_model")
    tokenizer.save_pretrained("models/sentiment_model")
    logger.info("Model saved to models/sentiment_model")
