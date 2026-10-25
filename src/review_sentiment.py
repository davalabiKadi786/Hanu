"""Score an existing review CSV with VADER and simple aspect tags."""
import argparse
import hashlib
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ASPECTS = {
    "Good Design & Quality": ["design", "quality", "build", "colour", "color", "material"],
    "Performance & Function": ["works", "working", "performance", "fast", "easy", "function"],
    "Value & Price": ["price", "value", "cost", "worth", "expensive"],
    "Reliability & Defect": ["broken", "defect", "stopped", "failed", "issue", "return"],
    "Service & Delivery": ["delivery", "support", "service", "shipping", "help"],
}


def score(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)
    if "title" not in df or "body" not in df:
        raise ValueError("Input must contain title and body columns")
    analyzer = SentimentIntensityAnalyzer()
    title_text = df["title"].fillna("").astype(str)
    body_text = df["body"].fillna("").astype(str)
    df["review_text"] = title_text.str.cat(body_text, sep=" ").str.strip()
    def label(text: str) -> str:
        compound = analyzer.polarity_scores(text)["compound"]
        return "Positive" if compound >= 0.05 else "Negative" if compound <= -0.05 else "Neutral"
    df["sentiment"] = df.review_text.map(label)
    def tags(text: str, sentiment: str) -> list[str]:
        text = text.lower()
        found = [name + " (" + sentiment[:3] + ")" for name, words in ASPECTS.items() if any(w in text for w in words)]
        return found or ["General Experience (" + sentiment[:3] + ")"]
    df["aspect_sentiment_tags"] = [tags(text, sentiment) for text, sentiment in zip(df.review_text, df.sentiment)]
    if "id" not in df:
        df["id"] = ""
    df["id"] = df["id"].fillna("").astype(str)
    needs_id = df["id"].eq("") | df["id"].duplicated(keep="first")
    for i in df.index[needs_id]:
        df.at[i, "id"] = hashlib.sha256(f"{i}|{df.at[i, 'review_text']}".encode("utf-8")).hexdigest()[:16]
    for col, default in [("source", "provided review dataset"), ("reviewer", "")]:
        if col not in df:
            df[col] = default
    if "date" in df:
        parsed = pd.to_datetime(df.date, errors="coerce")
        invalid = df.date.notna() & parsed.isna()
        if invalid.any():
            raise ValueError(f"{int(invalid.sum())} review dates could not be parsed; fix them before scoring")
        df["date"] = parsed.dt.strftime("%Y-%m-%d")
    else:
        df["date"] = ""
    if "rating" in df:
        ratings = pd.to_numeric(df.rating, errors="coerce")
        has_value = df.rating.notna() & df.rating.astype(str).str.strip().ne("")
        invalid = (has_value & ratings.isna()) | (ratings.notna() & ~ratings.between(1, 5))
        if invalid.any():
            raise ValueError("Ratings must be between 1 and 5")
        df["rating"] = ratings
    else:
        df["rating"] = pd.NA
    required = ["id", "title", "body", "date", "rating", "source", "reviewer", "sentiment", "aspect_sentiment_tags"]
    df = df[required + [c for c in df.columns if c not in required]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df):,} scored reviews to {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    score(a.input, a.output)

