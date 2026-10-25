"""Streamlit dashboard for exploring and mining the vehicle complaint dataset."""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.text_mining import data_quality, enrich_text, parse_list_cell

st.set_page_config(page_title="HanuAi ML Assessment", layout="wide")
st.title("HanuAi ML Assessment")
st.caption("Vehicle complaint exploration, entity tagging, and issue clustering")

uploaded = st.sidebar.file_uploader("Choose a complaint CSV", type=["csv"])
default_path = Path("data/raw/Dataset.csv")
if uploaded is not None:
    raw = pd.read_csv(uploaded, low_memory=False)
elif default_path.exists():
    raw = pd.read_csv(default_path, low_memory=False)
else:
    st.info("Upload the provided Dataset.csv or place it at data/raw/Dataset.csv.")
    st.stop()

clusters = st.sidebar.slider("Number of complaint clusters", 2, 12, 6)
with st.spinner("Profiling and mining complaint text..."):
    enriched, cluster_terms = enrich_text(raw, clusters)

q = data_quality(raw)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", f"{q['rows']:,}")
c2.metric("Columns", q["columns"])
c3.metric("Exact duplicate rows", q["exact_duplicate_rows"])
c4.metric("Duplicate event IDs", q["duplicate_event_ids"] if q["duplicate_event_ids"] is not None else "N/A")

tab_summary, tab_quality, tab_records, tab_cluster = st.tabs(["Summary", "Data quality", "Enriched records", "Clusters"])
with tab_summary:
    left, right = st.columns(2)
    with left:
        st.subheader("Mined issue categories")
        st.bar_chart(enriched.issue_category.value_counts())
    with right:
        st.subheader("Top structured failure components")
        if "Failure Component" in raw:
            components = raw["Failure Component"].map(parse_list_cell).explode().dropna()
            st.bar_chart(components.value_counts().head(12))
    if "Opened date" in raw:
        timeline = pd.to_datetime(raw["Opened date"], errors="coerce").dt.to_period("M").value_counts().sort_index()
        st.subheader("Events by opened month")
        st.line_chart(timeline)
    st.caption("Tags and clusters are automated candidate signals for analyst review, not confirmed root causes.")
with tab_quality:
    st.subheader("Schema and missingness")
    profile = pd.DataFrame({"dtype": q["dtypes"], "missing": q["missing_by_column"]})
    st.dataframe(profile, use_container_width=True)
    st.json({"exact_duplicate_rows": q["exact_duplicate_rows"], "duplicate_event_ids": q["duplicate_event_ids"]})
with tab_records:
    categories = ["All"] + sorted(enriched.issue_category.unique().tolist())
    chosen = st.selectbox("Issue category", categories)
    view = enriched if chosen == "All" else enriched[enriched.issue_category == chosen]
    st.dataframe(view.head(500), use_container_width=True, height=520)
    st.download_button("Download enriched CSV", enriched.to_csv(index=False).encode("utf-8-sig"), "enriched_dataset.csv", "text/csv")
with tab_cluster:
    st.subheader("Cluster descriptions")
    cluster_rows = [{"cluster": int(k), "top_terms": ", ".join(v), "records": int((enriched.cluster == int(k)).sum())} for k, v in cluster_terms.items()]
    st.dataframe(pd.DataFrame(cluster_rows), use_container_width=True)
    selected = st.selectbox("Inspect a cluster", sorted(enriched.cluster.unique().tolist()))
    cols = [x for x in ["Event id", "MODEL", "CUSTOMER_VERBATIM", "CAUSAL_VERBATIM", "issue_category"] if x in enriched]
    st.dataframe(enriched.loc[enriched.cluster == selected, cols].head(50), use_container_width=True)

