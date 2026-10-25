"""Explainable entity tagging and complaint clustering for the supplied data."""
from __future__ import annotations

import ast
import re

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

TEXT_COLUMNS = ["CUSTOMER_VERBATIM", "CAUSAL_VERBATIM", "CORRECTION_VERBATIM"]
COMPONENT_TERMS = ["radio", "display", "screen", "audio", "amplifier", "speaker", "camera", "wiring", "connector", "harness", "module", "battery", "antenna", "ethernet", "usb", "wifi", "onstar"]
SYMPTOM_TERMS = ["black screen", "blank", "no sound", "inoperative", "intermittent", "reset", "cut out", "no audio", "lost communication", "not working", "flicker", "disconnect", "inop"]
REPAIR_TERMS = ["replaced", "programmed", "reprogrammed", "repaired", "updated", "reset", "rewired", "reconnected"]


def combine_text(row: pd.Series) -> str:
    """Combine narrative fields while tolerating nulls and non-string values."""
    return " ".join(str(row.get(col, "")) for col in TEXT_COLUMNS if pd.notna(row.get(col, ""))).strip()


def clean_text(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^\w\s#-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_list_cell(value: object) -> list[str]:
    """Parse Python-list-like tag values safely; malformed values become empty."""
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
        return [str(x).strip() for x in parsed] if isinstance(parsed, (list, tuple)) else [str(parsed)]
    except (ValueError, SyntaxError):
        return []


def find_terms(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text)]


def classify_issue(text: str, components: list[str], symptoms: list[str]) -> str:
    t = text.lower()
    if any(x in t for x in ["ethernet", "communication", "dtc", "network", "bus"]):
        return "Communication / Network"
    if any(x in t for x in ["short", "ground", "voltage", "power", "fuse", "electrical", "circuit"]):
        return "Electrical / Power"
    if any(x in t for x in ["software", "program", "update", "firmware", "reset"]):
        return "Software / Programming"
    if any(x in t for x in ["water", "corrosion", "rust", "moisture"]):
        return "Environmental / Corrosion"
    if any(x in t for x in ["loose", "fitment", "assembly", "rattle", "broken", "physical damage"]):
        return "Assembly / Physical"
    if components or symptoms:
        return "Component / Functional"
    return "Other / Unspecified"


def enrich_text(df: pd.DataFrame, cluster_count: int = 6) -> tuple[pd.DataFrame, dict]:
    result = df.copy()
    for col in TEXT_COLUMNS:
        if col not in result:
            result[col] = ""
    result["analysis_text"] = result.apply(combine_text, axis=1)
    result["cleaned_text"] = result["analysis_text"].map(clean_text)
    result["mined_components"] = result["cleaned_text"].map(lambda x: find_terms(x, COMPONENT_TERMS))
    result["mined_symptoms"] = result["cleaned_text"].map(lambda x: find_terms(x, SYMPTOM_TERMS))
    result["mined_repairs"] = result["cleaned_text"].map(lambda x: find_terms(x, REPAIR_TERMS))
    result["issue_category"] = [classify_issue(t, c, s) for t, c, s in zip(result.cleaned_text, result.mined_components, result.mined_symptoms)]

    # Fit only on meaningful narratives. Empty text records retain cluster -1.
    result["cluster"] = -1
    valid = result.cleaned_text.str.len() >= 8
    n = int(valid.sum())
    terms_by_cluster: dict[str, list[str]] = {}
    if n:
        k = max(1, min(int(cluster_count), n))
        vectorizer = TfidfVectorizer(stop_words="english", max_features=2500, ngram_range=(1, 2), min_df=2)
        try:
            matrix = vectorizer.fit_transform(result.loc[valid, "cleaned_text"])
        except ValueError as exc:
            # A corpus made only of stop words or one-character tokens has no vocabulary.
            if "empty vocabulary" not in str(exc).lower():
                raise
            result.loc[valid, "cluster"] = 0
            terms_by_cluster["0"] = []
            return result, terms_by_cluster
        if matrix.shape[1] == 0:
            result.loc[valid, "cluster"] = 0
            terms_by_cluster["0"] = []
        else:
            k = min(k, matrix.shape[0], matrix.shape[1])
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(matrix)
            result.loc[valid, "cluster"] = labels
            vocab = vectorizer.get_feature_names_out()
            for cluster_id, center in enumerate(model.cluster_centers_):
                terms_by_cluster[str(cluster_id)] = vocab[center.argsort()[-10:][::-1]].tolist()
    return result, terms_by_cluster


def data_quality(df: pd.DataFrame) -> dict:
    # pandas duplicate detection requires hashable cells; normalize nested lists safely.
    comparable = df.copy()
    for col in comparable.columns:
        if comparable[col].map(lambda value: isinstance(value, (list, dict, set, tuple))).any():
            comparable[col] = comparable[col].map(lambda value: repr(value) if isinstance(value, (list, dict, set, tuple)) else value)
    return {
        "rows": int(len(df)), "columns": int(len(df.columns)),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "missing_by_column": {str(k): int(v) for k, v in df.isna().sum().items()},
        "exact_duplicate_rows": int(comparable.duplicated().sum()),
        "duplicate_event_ids": int(df["Event id"].duplicated().sum()) if "Event id" in df else None,
    }
