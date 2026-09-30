"""Streamlit interface for the business entity resolution workbench."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.data_generator import DATASET_SIZE, SEED, generate_dataset
from src.matching import (
    candidate_feature_frame,
    evaluate_test_set,
    generate_candidate_pairs,
    load_model,
    load_split,
    normalize_text,
    predict_for_reference,
    train_model,
)

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
MODEL_DIR = APP_DIR / "models"
PAGES = [
    "Dashboard", "Dataset", "Preprocessing", "Candidate Generation",
    "Model Training", "Evaluation", "Entity Matching", "Results", "Reports",
]

st.set_page_config(
    page_title="Business Entity Resolution",
    page_icon="B",
    layout="wide",
    initial_sidebar_state="expanded",
)


def ensure_data() -> dict[str, int] | None:
    required = [
        DATA_DIR / split / f"{split}_{source}.tsv"
        for split in ("train", "test")
        for source in ("source1", "source2", "source3")
    ]
    required += [DATA_DIR / split / f"{split}_ground_truth.tsv" for split in ("train", "test")]
    if all(path.exists() for path in required):
        return None
    return generate_dataset(DATA_DIR, DATASET_SIZE, SEED)


@st.cache_data(show_spinner=False)
def cached_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_split(DATA_DIR, split)


@st.cache_data(show_spinner=False)
def cached_evaluation(data_root: str, _bundle: dict[str, Any], model_stamp: float, threshold: float):
    del model_stamp
    return evaluate_test_set(data_root, _bundle, threshold)


def get_bundle() -> dict[str, Any] | None:
    return load_model(MODEL_DIR)


def metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def show_sidebar() -> str:
    bundle = get_bundle()
    with st.sidebar:
        st.title("Entity Resolution")
        st.caption("ML-based business matching system")
        page = st.radio("Workspace", PAGES, label_visibility="collapsed")
        st.divider()
        if bundle:
            st.success(f"Model ready · threshold {bundle['threshold']:.3f}")
        else:
            st.info("Train a model to score business matches.")
        st.caption("Synthetic records · no external data services")
    return page


def show_dashboard() -> None:
    reference, source2, source3, truth = cached_split("train")
    test_ref, test2, test3, test_truth = cached_split("test")
    bundle = get_bundle()
    st.title("Business Entity Resolution")
    st.subheader("ML-Based Business Matching System")
    st.write(
        "Find records that describe the same business across independent registries. "
        "Candidate blocking keeps comparisons practical; a supervised model ranks likely matches."
    )
    cols = st.columns(4)
    total = len(reference) + len(source2) + len(source3) + len(test_ref) + len(test2) + len(test3)
    cols[0].metric("Synthetic records", f"{total:,}", "All three sources, train and test")
    cols[1].metric("Training businesses", f"{reference['entity_id'].nunique():,}")
    cols[2].metric("Test match pairs", f"{len(test_truth):,}")
    cols[3].metric("Model status", "Ready" if bundle else "Not trained")
    st.divider()

    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("Registry coverage")
        chart = pd.DataFrame({
            "Training": [len(reference), len(source2), len(source3)],
            "Test": [len(test_ref), len(test2), len(test3)],
        }, index=["Source 1 · reference", "Source 2", "Source 3"])
        st.bar_chart(chart, horizontal=True)
    with right:
        st.subheader("How the workflow runs")
        st.markdown(
            "1. **Prepare** noisy names and addresses using consistent normalization.\n"
            "2. **Generate candidates** with country blocking and character n-gram retrieval.\n"
            "3. **Train** a Random Forest on labeled candidate pairs.\n"
            "4. **Evaluate** against a separate, entity-disjoint test set.\n"
            "5. **Match** new reference records into both source registries."
        )
        if bundle:
            metrics = bundle.get("validation_metrics", {})
            st.info(
                f"Current operating threshold: **{bundle['threshold']:.3f}** · "
                f"validation precision: **{metrics.get('precision', 0):.1%}**"
            )
        else:
            st.warning("Start in **Model Training** to create a local model before matching.")

    st.subheader("Data quality built into the demo")
    st.write(
        "Business names and addresses vary through spelling changes, abbreviations, punctuation, "
        "case, legal suffixes, partial locations, reordered components, missing records, duplicates, "
        "and similar-but-unrelated names. Each business is assigned wholly to train or test."
    )
    st.caption(
        f"Training includes {len(truth):,} labeled positive pairs across Source 1 → Source 2 "
        "and Source 1 → Source 3. Labels are not model features."
    )


def show_dataset() -> None:
    train_ref, train2, train3, train_truth = cached_split("train")
    test_ref, test2, test3, test_truth = cached_split("test")
    st.title("Dataset")
    st.write("Explore the bundled TSV registries. Ground-truth entity IDs are present for reproducibility and evaluation only.")
    selected = st.selectbox("Data split", ["Training", "Test"])
    if selected == "Training":
        reference, source2, source3, truth = train_ref, train2, train3, train_truth
    else:
        reference, source2, source3, truth = test_ref, test2, test3, test_truth
    counts = st.columns(4)
    counts[0].metric("Source 1", f"{len(reference):,}")
    counts[1].metric("Source 2", f"{len(source2):,}")
    counts[2].metric("Source 3", f"{len(source3):,}")
    counts[3].metric("Labeled match pairs", f"{len(truth):,}")
    tab1, tab2, tab3, tab4 = st.tabs(["Source 1", "Source 2", "Source 3", "Ground truth"])
    for tab, data, filename in (
        (tab1, reference, "source1"), (tab2, source2, "source2"),
        (tab3, source3, "source3"), (tab4, truth, "ground_truth"),
    ):
        with tab:
            display = data.drop(columns=["entity_id"], errors="ignore")
            st.dataframe(display.head(100), use_container_width=True, hide_index=True)
            st.caption(f"{len(data):,} records in {selected.lower()} split.")
            st.download_button(
                f"Download {filename} TSV",
                data=data.to_csv(sep="\t", index=False).encode("utf-8"),
                file_name=f"{selected.lower()}_{filename}.tsv",
                mime="text/tab-separated-values",
                key=f"download_{selected}_{filename}",
            )
    country_counts = pd.concat([reference["country"], source2["country"], source3["country"]]).value_counts()
    st.subheader("Records by country")
    st.bar_chart(country_counts)
    st.caption("Countries represented: India, USA, and France.")


def show_preprocessing() -> None:
    reference, source2, source3, _ = cached_split("train")
    st.title("Preprocessing")
    st.write(
        "The matching pipeline folds case and accents, removes punctuation and common legal suffixes, "
        "expands common street abbreviations, and normalizes whitespace. Original fields remain intact."
    )
    source_choice = st.selectbox("Preview registry", ["Source 1", "Source 2", "Source 3"])
    selected_frame = {"Source 1": reference, "Source 2": source2, "Source 3": source3}[source_choice]
    row_index = st.number_input("Record row", min_value=0, max_value=max(0, len(selected_frame) - 1), value=0)
    row = selected_frame.iloc[int(row_index)]
    st.dataframe(pd.DataFrame([
        {"Field": "Business name", "Original": row["business_name"], "Normalized": normalize_text(row["business_name"])},
        {"Field": "Business address", "Original": row["business_address"], "Normalized": normalize_text(row["business_address"])},
        {"Field": "Country", "Original": row["country"], "Normalized": normalize_text(row["country"])},
    ]), use_container_width=True, hide_index=True)
    st.subheader("Pairwise comparison signals")
    st.write(
        "Each candidate pair is described by name/address sequence and token similarity, "
        "character n-gram cosine similarity, exact and containment checks, house-number agreement, "
        "relative text lengths, country agreement, and source channel."
    )
    st.info("The synthetic entity ID is used only as a training/evaluation label. It is excluded from all model features.")


def show_candidate_generation() -> None:
    reference, source2, source3, _ = cached_split("train")
    st.title("Candidate Generation")
    st.write(
        "Instead of scoring every possible pair, the first stage retrieves a short list of likely candidates "
        "inside the same country, using character n-grams on both names and addresses."
    )
    index = st.number_input("Source 1 row", min_value=0, max_value=max(0, len(reference) - 1), value=0)
    ref = reference.iloc[[int(index)]].reset_index(drop=True)
    st.caption(f"Reference: {ref.iloc[0]['business_name']} · {ref.iloc[0]['business_address']} · {ref.iloc[0]['country']}")
    source_choice = st.radio("Compare against", ["Source 2", "Source 3"], horizontal=True)
    candidate_source = source2 if source_choice == "Source 2" else source3
    with st.spinner("Retrieving name and address neighbours..."):
        pairs = generate_candidate_pairs(ref, candidate_source, top_k=12)
    if not pairs:
        st.warning("No candidates were retrieved for this country.")
        return
    features = candidate_feature_frame(
        ref, candidate_source, pairs, 0 if source_choice == "Source 2" else 1
    )
    result = features.merge(
        candidate_source[["record_id", "business_name", "business_address", "country"]],
        left_on="matched_record_id", right_on="record_id", how="left",
    )
    result["candidate_score"] = result[["name_vector_cosine", "address_vector_cosine"]].mean(axis=1)
    result = result.sort_values("candidate_score", ascending=False)
    st.metric("Candidates retrieved", f"{len(result):,}", "Union of name and address neighbours")
    st.dataframe(result[[
        "matched_record_id", "business_name", "business_address", "country",
        "name_vector_cosine", "address_vector_cosine", "candidate_score",
    ]].head(25), use_container_width=True, hide_index=True)
    st.caption("Candidate scores help inspect retrieval; they are not final match probabilities.")


def show_model_training() -> None:
    st.title("Model Training")
    st.write(
        "Train a Random Forest classifier on generated candidate pairs. A held-out portion of training entities "
        "sets a conservative confidence threshold; final evaluation uses the independent test split."
    )
    reference, source2, source3, truth = cached_split("train")
    col1, col2, col3 = st.columns(3)
    col1.metric("Training reference records", f"{len(reference):,}")
    col2.metric("Source 2 + Source 3 records", f"{len(source2) + len(source3):,}")
    col3.metric("Positive training pairs", f"{len(truth):,}")
    bundle = get_bundle()
    if bundle:
        st.success(f"Model trained {bundle.get('trained_at', 'previously')}.")
        st.metric("Stored precision-focused threshold", f"{bundle['threshold']:.3f}")
        st.json(bundle.get("validation_metrics", {}))
    if st.button("Train / retrain model", type="primary"):
        try:
            with st.spinner("Generating training candidates and fitting the model. This may take a minute..."):
                bundle = train_model(DATA_DIR, MODEL_DIR)
            st.success("Training completed. The model is saved locally and ready for matching.")
            st.json({
                "threshold": bundle["threshold"],
                "validation_metrics": bundle["validation_metrics"],
                "training_rows": bundle["training_rows"],
                "training_positive_pairs": bundle["training_positive_pairs"],
                "training_negative_pairs": bundle["training_negative_pairs"],
            })
            st.rerun()
        except Exception as error:
            st.error(f"Training failed: {error}")


def get_evaluation(bundle: dict[str, Any], threshold: float):
    model_path = MODEL_DIR / "entity_resolution.joblib"
    return cached_evaluation(str(DATA_DIR), bundle, model_path.stat().st_mtime, threshold)


def show_evaluation() -> tuple[dict[str, Any] | None, pd.DataFrame | None]:
    st.title("Evaluation")
    bundle = get_bundle()
    if not bundle:
        st.warning("Train a model on the Model Training page before running test-set evaluation.")
        return None, None
    threshold = st.slider(
        "Match confidence threshold",
        min_value=0.5, max_value=0.999, value=float(bundle["threshold"]), step=0.001,
        help="Increasing the threshold normally improves precision and reduces recall.",
    )
    with st.spinner("Evaluating against the entity-disjoint test set..."):
        summary, pairs = get_evaluation(bundle, threshold)
    metrics = summary["metrics"]
    cols = st.columns(4)
    cols[0].metric("Precision", f"{metrics['precision']:.1%}")
    cols[1].metric("Recall", f"{metrics['recall']:.1%}")
    cols[2].metric("F1", f"{metrics['f1']:.1%}")
    cols[3].metric("Candidate recall", f"{metrics['candidate_recall']:.1%}")
    st.caption(
        f"Test pairs: {summary['ground_truth_pairs']:,} ground-truth matches · "
        f"{metrics['predicted_matches']:,} predicted matches · threshold {threshold:.3f}."
    )
    st.subheader("Results by source")
    by_source = pd.DataFrame(summary["by_source"]).T
    st.dataframe(by_source[["precision", "recall", "f1", "candidate_recall", "predicted_matches"]], use_container_width=True)
    st.subheader("Model score distribution")
    if not pairs.empty:
        scored = pairs.copy()
        scored["Outcome"] = scored.apply(
            lambda row: "True match" if row["label"] and row["match_probability"] >= threshold
            else "Missed match" if row["label"]
            else "Predicted match" if row["match_probability"] >= threshold
            else "Other candidate", axis=1,
        )
        st.bar_chart(scored.groupby("Outcome")["match_probability"].count())
    st.download_button(
        "Download evaluation summary JSON",
        data=json.dumps(summary, indent=2).encode("utf-8"),
        file_name="evaluation_summary.json",
        mime="application/json",
    )
    return summary, pairs


def show_entity_matching() -> None:
    st.title("Entity Matching")
    st.write("Enter a reference business to find likely records in Source 2 and Source 3.")
    bundle = get_bundle()
    if not bundle:
        st.warning("Train a model first. Matching uses the trained model and its confidence threshold.")
        return
    with st.form("matching_form"):
        name = st.text_input("Business name", placeholder="Sri Lakshmi Textiles Pvt Ltd")
        address = st.text_input("Business address", placeholder="123 Gandhi Road, Coimbatore")
        country = st.selectbox("Country", ["India", "USA", "France"])
        threshold = st.slider(
            "Confidence threshold", min_value=0.5, max_value=0.999,
            value=float(bundle["threshold"]), step=0.001,
        )
        submitted = st.form_submit_button("Find matches", type="primary")
    if submitted:
        if not name.strip() or not address.strip():
            st.error("Enter both a business name and an address.")
        else:
            query = pd.DataFrame([{
                "record_id": "MANUAL-QUERY",
                "entity_id": "",
                "business_name": name.strip(),
                "business_address": address.strip(),
                "country": country,
            }])
            _, source2, source3, _ = cached_split("train")
            all_results = []
            with st.spinner("Scoring candidates in both registries..."):
                for source, candidates, channel in (("source2", source2, 0), ("source3", source3, 1)):
                    matches = predict_for_reference(query, candidates, channel, bundle, threshold)
                    if not matches.empty:
                        matches["matched_source"] = source
                        all_results.append(matches)
            result = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
            st.session_state["last_match_query"] = query.iloc[0].to_dict()
            st.session_state["last_match_results"] = result
            st.session_state["last_match_threshold"] = threshold
            if result.empty:
                st.info("No candidates met the confidence threshold. Try a lower threshold only if higher recall is more important than precision.")
            else:
                st.success(f"Found {len(result)} high-confidence candidate match(es).")
                shown = result.rename(columns={
                    "matched_source": "Source",
                    "match_probability": "Confidence",
                    "matched_record_id": "Record ID",
                    "business_name": "Business name",
                    "business_address": "Business address",
                    "country": "Country",
                })
                shown["Confidence"] = shown["Confidence"].map(lambda value: f"{value:.1%}")
                st.dataframe(shown[[
                    "Source", "Record ID", "Business name", "Business address", "Country", "Confidence"
                ]], use_container_width=True, hide_index=True)


def show_results() -> None:
    st.title("Results")
    results = st.session_state.get("last_match_results")
    query = st.session_state.get("last_match_query")
    if query is None:
        st.info("Run a query in Entity Matching. Its scored matches will be available here for review and download.")
        return
    st.write(f"Latest query: **{query['business_name']}** · {query['business_address']} · {query['country']}")
    if results is None or results.empty:
        st.warning("The latest query has no results above its confidence threshold.")
        return
    st.metric("Returned match candidates", f"{len(results):,}")
    st.dataframe(results[[
        "matched_source", "matched_record_id", "business_name", "business_address",
        "country", "match_probability",
    ]], use_container_width=True, hide_index=True)
    st.download_button(
        "Download latest matching results TSV",
        data=results.to_csv(sep="\t", index=False).encode("utf-8"),
        file_name="entity_match_results.tsv",
        mime="text/tab-separated-values",
    )


def show_reports() -> None:
    st.title("Reports")
    bundle = get_bundle()
    if not bundle:
        st.info("Train a model, then run Evaluation to create a report for the test set.")
        st.download_button(
            "Download dataset-generation summary",
            data=json.dumps({"configured_dataset_size": DATASET_SIZE, "seed": SEED}, indent=2).encode("utf-8"),
            file_name="dataset_summary.json",
            mime="application/json",
        )
        return
    threshold = st.slider(
        "Report confidence threshold", 0.5, 0.999, float(bundle["threshold"]), 0.001,
        key="report_threshold",
    )
    try:
        with st.spinner("Preparing test-set report..."):
            summary, pairs = get_evaluation(bundle, threshold)
        metrics = summary["metrics"]
        st.subheader("Test-set report")
        st.write(
            f"On {summary['ground_truth_pairs']:,} labeled pairs, the current model predicts "
            f"{metrics['predicted_matches']:,} matches at threshold {threshold:.3f}. "
            f"Precision is {metrics['precision']:.1%}; recall is {metrics['recall']:.1%}."
        )
        left, right = st.columns(2)
        with left:
            st.download_button(
                "Download JSON metrics",
                data=json.dumps(summary, indent=2).encode("utf-8"),
                file_name="entity_resolution_report.json",
                mime="application/json",
            )
        with right:
            output = pairs[pairs["match_probability"] >= threshold].copy() if not pairs.empty else pairs
            st.download_button(
                "Download predicted test pairs TSV",
                data=output.to_csv(sep="\t", index=False).encode("utf-8"),
                file_name="predicted_test_pairs.tsv",
                mime="text/tab-separated-values",
            )
        with st.expander("View report metrics"):
            st.json(summary)
    except Exception as error:
        st.error(f"Could not prepare the evaluation report: {error}")


def main() -> None:
    generated = ensure_data()
    if generated:
        st.toast(f"Generated {generated['total_records']:,} synthetic records.")
    page = show_sidebar()
    pages = {
        "Dashboard": show_dashboard,
        "Dataset": show_dataset,
        "Preprocessing": show_preprocessing,
        "Candidate Generation": show_candidate_generation,
        "Model Training": show_model_training,
        "Entity Matching": show_entity_matching,
        "Results": show_results,
        "Reports": show_reports,
    }
    if page == "Evaluation":
        show_evaluation()
    else:
        pages[page]()


if __name__ == "__main__":
    main()