"""Create a stakeholder summary from an authorized, sentiment-scored review CSV."""
import argparse
import ast
from pathlib import Path

import pandas as pd


def create_report(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path, low_memory=False)
    required = {"sentiment", "aspect_sentiment_tags"}
    if not required.issubset(df.columns):
        raise ValueError("Score the review CSV first with python -m src.review_sentiment")
    sentiment = df.sentiment.value_counts(dropna=False)
    def parse_tags(value: object) -> list[str]:
        if not isinstance(value, str) or not value.strip():
            return []
        try:
            parsed = ast.literal_eval(value)
            return [str(x) for x in parsed] if isinstance(parsed, (list, tuple)) else [str(parsed)]
        except (ValueError, SyntaxError):
            return [x.strip() for x in value.split("|") if x.strip()]
    tags = df.aspect_sentiment_tags.map(parse_tags).explode().dropna()
    top_tags = tags[tags.str.len() > 0].value_counts().head(12)
    negative = top_tags[top_tags.index.str.contains(r"\(Neg\)|\(Negat", case=False, regex=True)]
    negative_lines = "\n".join(f"- {name}: {count} tagged reviews" for name, count in negative.items()) or "- No negative aspect labels were detected. Manually review a sample."
    tag_lines = "\n".join(f"- {name}: {count}" for name, count in top_tags.items()) or "- No aspect tags available."
    counts = "\n".join(f"- {name}: {int(count)}" for name, count in sentiment.items())
    report = f'''# BestBuy Canada Review Sentiment Summary

## Coverage

Analyzed **{len(df):,} rows** from the review CSV supplied to the sentiment workflow. Confirm the selected product had at least 50 reviews and that the review page and sort coverage are documented before treating results as representative.

## Overall sentiment

{counts}

## Most common aspect-sentiment tags

{tag_lines}

## Dissatisfaction signals to review

{negative_lines}

## Recommendations

1. Read a sample of positive and negative reviews in each frequent aspect to verify the lexicon label and understand the customer's issue.
2. Prioritize recurring negative aspects for product and service investigation; verify return or repair data before estimating impact.
3. Compare sentiment across rating levels and review dates to identify mismatches and changes over time.
4. Record unique review counts, date range, page coverage, and any pages that could not be loaded.

## Method and access notes

- VADER baseline: positive at compound polarity 0.05 or above, negative at -0.05 or below, neutral between those values. Aspect tags use keyword rules.
- This is not a fine-tuned model; validate labels on a human-reviewed sample before using them for decisions.
- Best Buy Canada's Conditions of Use prohibit scraping and data mining. Automated collection must not be run unless written permission specifically authorizes it. Do not bypass CAPTCHA, bot controls, IP restrictions, or other access controls. Use an approved export or authorized data source when permission is unavailable.
'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/task1_stakeholder_report.md"))
    args = parser.parse_args()
    create_report(args.input, args.output)
