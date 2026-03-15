import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI
from pydantic import BaseModel
import re

app = FastAPI(title="Sentiment Analysis API", version="1.0")

MODEL_PATH = "models/sentiment_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

ID2LABEL = model.config.id2label


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TextRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    scores: dict


@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(req: TextRequest):
    cleaned = clean_text(req.text)
    inputs = tokenizer(
        cleaned, return_tensors="pt",
        truncation=True, padding=True, max_length=128
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_id = int(probs.argmax())
    return SentimentResponse(
        text=cleaned,
        sentiment=ID2LABEL[pred_id],
        confidence=float(probs[pred_id]),
        scores={ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
