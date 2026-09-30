"""Candidate generation, ML training, evaluation, and inference."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors

FEATURE_COLUMNS = [
    "name_sequence_ratio", "address_sequence_ratio",
    "name_token_jaccard", "address_token_jaccard",
    "name_vector_cosine", "address_vector_cosine",
    "name_exact", "address_exact",
    "name_containment", "address_containment",
    "house_number_agreement", "name_length_ratio", "address_length_ratio",
    "name_token_count_ratio", "country_agreement", "source_channel",
]
MODEL_PATH_NAME = "entity_resolution.joblib"


def normalize_text(value: Any) -> str:
    """Fold typography and legal suffixes while retaining identifying words."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char)).lower()
    text = re.sub(r"\b(private limited|private ltd|pvt limited|pvt ltd|pvt|limited|ltd|incorporated|inc|llc|sarl|sas|gmbh|corporation|corp)\b", " ", text)
    text = re.sub(r"\b(rd|st|ave|blvd)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = []
    for token in text.split():
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return " ".join(tokens)


def _token_jaccard(left: str, right: str) -> float:
    a, b = set(left.split()), set(right.split())
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _safe_ratio(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def pair_features(
    reference: pd.Series | dict[str, Any],
    candidate: pd.Series | dict[str, Any],
    name_cosine: float,
    address_cosine: float,
    source_channel: int,
) -> dict[str, float]:
    name_a = normalize_text(reference.get("business_name", ""))
    name_b = normalize_text(candidate.get("business_name", ""))
    address_a = normalize_text(reference.get("business_address", ""))
    address_b = normalize_text(candidate.get("business_address", ""))
    name_tokens_a, name_tokens_b = name_a.split(), name_b.split()
    address_tokens_a, address_tokens_b = address_a.split(), address_b.split()
    house_a = next((x for x in address_tokens_a if x.isdigit()), "")
    house_b = next((x for x in address_tokens_b if x.isdigit()), "")
    return {
        "name_sequence_ratio": _safe_ratio(name_a, name_b),
        "address_sequence_ratio": _safe_ratio(address_a, address_b),
        "name_token_jaccard": _token_jaccard(name_a, name_b),
        "address_token_jaccard": _token_jaccard(address_a, address_b),
        "name_vector_cosine": float(name_cosine),
        "address_vector_cosine": float(address_cosine),
        "name_exact": float(name_a == name_b),
        "address_exact": float(address_a == address_b),
        "name_containment": float(bool(name_a and name_b and (name_a in name_b or name_b in name_a))),
        "address_containment": float(bool(address_a and address_b and (address_a in address_b or address_b in address_a))),
        "house_number_agreement": float(not house_a and not house_b or bool(house_a and house_b and house_a == house_b)),
        "name_length_ratio": min(len(name_a), len(name_b)) / max(1, len(name_a), len(name_b)),
        "address_length_ratio": min(len(address_a), len(address_b)) / max(1, len(address_a), len(address_b)),
        "name_token_count_ratio": min(len(name_tokens_a), len(name_tokens_b)) / max(1, len(name_tokens_a), len(name_tokens_b)),
        "country_agreement": float(str(reference.get("country", "")).casefold() == str(candidate.get("country", "")).casefold()),
        "source_channel": float(source_channel),
    }


def generate_candidate_pairs(
    references: pd.DataFrame,
    candidates: pd.DataFrame,
    top_k: int = 12,
) -> list[dict[str, Any]]:
    """Block by country and union nearest neighbours on name and address."""
    if references.empty or candidates.empty:
        return []
    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    reference_countries = references["country"].fillna("").astype(str)
    candidate_countries = candidates["country"].fillna("").astype(str)
    for country in sorted(set(reference_countries) & set(candidate_countries)):
        ref_ids = np.flatnonzero(reference_countries.to_numpy() == country)
        cand_ids = np.flatnonzero(candidate_countries.to_numpy() == country)
        if not len(ref_ids) or not len(cand_ids):
            continue
        ref_block = references.iloc[ref_ids]
        cand_block = candidates.iloc[cand_ids]
        for field, score_key in (("business_name", "name_cosine"), ("business_address", "address_cosine")):
            ref_text = [normalize_text(x) for x in ref_block[field].fillna("")]
            cand_text = [normalize_text(x) for x in cand_block[field].fillna("")]
            vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 4), min_df=1,
                sublinear_tf=True, max_features=80_000,
            )
            matrix = vectorizer.fit_transform(ref_text + cand_text)
            ref_matrix, cand_matrix = matrix[: len(ref_ids)], matrix[len(ref_ids) :]
            k = min(top_k, len(cand_ids))
            nearest = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
            nearest.fit(cand_matrix)
            distances, indices = nearest.kneighbors(ref_matrix)
            for local_ref, (row_distances, row_indices) in enumerate(zip(distances, indices)):
                global_ref = int(ref_ids[local_ref])
                for distance, local_cand in zip(row_distances, row_indices):
                    global_cand = int(cand_ids[int(local_cand)])
                    key = (global_ref, global_cand)
                    score = float(max(0.0, 1.0 - distance))
                    entry = pairs.setdefault(key, {
                        "ref_index": global_ref,
                        "cand_index": global_cand,
                        "name_cosine": 0.0,
                        "address_cosine": 0.0,
                        "blocking_channels": 0,
                        "forced_positive": False,
                    })
                    entry[score_key] = max(entry[score_key], score)
                    entry["blocking_channels"] += 1
    return list(pairs.values())


def _truth_lookup(reference: pd.DataFrame, candidates: pd.DataFrame, truth: pd.DataFrame, source: str) -> set[tuple[int, int]]:
    if truth.empty:
        return set()
    ref_map = {value: index for index, value in enumerate(reference["record_id"].astype(str))}
    cand_map = {value: index for index, value in enumerate(candidates["record_id"].astype(str))}
    rows = truth[truth["matched_source"] == source]
    positives = set()
    for row in rows.itertuples(index=False):
        ref_idx = ref_map.get(str(row.source1_record_id))
        cand_idx = cand_map.get(str(row.matched_record_id))
        if ref_idx is not None and cand_idx is not None:
            positives.add((ref_idx, cand_idx))
    return positives


def candidate_feature_frame(
    references: pd.DataFrame,
    candidates: pd.DataFrame,
    candidate_pairs: list[dict[str, Any]],
    source_channel: int,
    positive_pairs: set[tuple[int, int]] | None = None,
    max_negatives_per_reference: int | None = None,
    include_missing_positives: bool = False,
) -> pd.DataFrame:
    positive_pairs = positive_pairs or set()
    by_reference: dict[int, list[dict[str, Any]]] = {}
    for pair in candidate_pairs:
        by_reference.setdefault(pair["ref_index"], []).append(pair)
    if include_missing_positives:
        existing_pairs = {
            (pair["ref_index"], pair["cand_index"]) for pair in candidate_pairs
        }
        for ref_idx, cand_idx in positive_pairs:
            if (ref_idx, cand_idx) not in existing_pairs:
                candidate_pairs.append({
                    "ref_index": ref_idx, "cand_index": cand_idx,
                    "name_cosine": 0.0, "address_cosine": 0.0,
                    "blocking_channels": 0, "forced_positive": True,
                })
                by_reference.setdefault(ref_idx, []).append(candidate_pairs[-1])
                existing_pairs.add((ref_idx, cand_idx))

    rows: list[dict[str, Any]] = []
    for ref_idx, ref_pairs in by_reference.items():
        ref_pairs.sort(
            key=lambda p: p["name_cosine"] + p["address_cosine"],
            reverse=True,
        )
        positives = [p for p in ref_pairs if (p["ref_index"], p["cand_index"]) in positive_pairs]
        negatives = [p for p in ref_pairs if (p["ref_index"], p["cand_index"]) not in positive_pairs]
        if max_negatives_per_reference is not None:
            negatives = negatives[:max_negatives_per_reference]
        for pair in positives + negatives:
            ref_row = references.iloc[pair["ref_index"]]
            cand_row = candidates.iloc[pair["cand_index"]]
            rows.append({
                "ref_index": pair["ref_index"],
                "cand_index": pair["cand_index"],
                "reference_entity_id": str(ref_row.get("entity_id", "")),
                "source1_record_id": str(ref_row["record_id"]),
                "matched_record_id": str(cand_row["record_id"]),
                "entity_id": str(ref_row.get("entity_id", "")),
                "forced_positive": bool(pair.get("forced_positive", False)),
                "label": int((pair["ref_index"], pair["cand_index"]) in positive_pairs),
                **pair_features(
                    ref_row, cand_row, pair["name_cosine"], pair["address_cosine"], source_channel
                ),
            })
    return pd.DataFrame(rows)


def _threshold_at_precision(y_true: np.ndarray, probabilities: np.ndarray, target_precision: float = 0.99) -> float:
    if not len(y_true) or len(set(y_true.tolist())) < 2:
        return 0.92
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    eligible = np.flatnonzero(precision[:-1] >= target_precision)
    if len(eligible):
        best = eligible[np.argmax(recall[eligible])]
        return float(np.clip(thresholds[best], 0.5, 0.995))
    # If the calibration fold cannot reach the requested precision, prefer the
    # strongest observed operating point and keep the decision conservative.
    best_precision = float(np.max(precision[:-1])) if len(thresholds) else 0.0
    if best_precision > 0 and len(thresholds):
        best = int(np.flatnonzero(precision[:-1] == best_precision)[-1])
        return float(np.clip(max(thresholds[best], 0.92), 0.5, 0.999))
    return 0.92


def _evaluate_predictions(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float, truth_count: int | None = None) -> dict[str, float | int]:
    y_true = frame["label"].to_numpy(dtype=int)
    predicted = probabilities >= threshold
    tp = int(np.sum(predicted & (y_true == 1)))
    fp = int(np.sum(predicted & (y_true == 0)))
    fn_candidates = int(np.sum(~predicted & (y_true == 1)))
    total_truth = max(truth_count or int(np.sum(y_true == 1)), int(np.sum(y_true == 1)))
    missed_blocking = max(0, total_truth - int(np.sum(y_true == 1)))
    fn = fn_candidates + missed_blocking
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "candidate_recall": int(np.sum(y_true == 1)) / max(1, total_truth),
        "predicted_matches": int(np.sum(predicted)),
    }


def _source_data(root: Path, split: str, source: str) -> pd.DataFrame:
    path = root / split / f"{split}_{source}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


def load_split(root: str | Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root_path = Path(root)
    reference = _source_data(root_path, split, "source1")
    source2 = _source_data(root_path, split, "source2")
    source3 = _source_data(root_path, split, "source3")
    truth_path = root_path / split / f"{split}_ground_truth.tsv"
    truth = pd.read_csv(truth_path, sep="\t", dtype=str).fillna("") if truth_path.exists() else pd.DataFrame(
        columns=["entity_id", "matched_source", "source1_record_id", "matched_record_id"]
    )
    return reference, source2, source3, truth


def _assemble_split_pairs(
    reference: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    truth: pd.DataFrame,
    top_k: int,
    training: bool,
    max_negatives_per_reference: int,
) -> pd.DataFrame:
    frames = []
    for source, candidates, channel in (("source2", source2, 0), ("source3", source3, 1)):
        pairs = generate_candidate_pairs(reference, candidates, top_k=top_k)
        positives = _truth_lookup(reference, candidates, truth, source)
        frame = candidate_feature_frame(
            reference, candidates, pairs, channel, positive_pairs=positives,
            max_negatives_per_reference=max_negatives_per_reference if training else None,
            include_missing_positives=training,
        )
        if not training and not frame.empty:
            frame = frame[~frame["forced_positive"]].copy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def train_model(data_dir: str | Path, model_dir: str | Path) -> dict[str, Any]:
    """Train a forest, calibrate its high-precision threshold, and persist it."""
    data_root, output = Path(data_dir), Path(model_dir)
    references, source2, source3, truth = load_split(data_root, "train")
    frame = _assemble_split_pairs(references, source2, source3, truth, top_k=12, training=True, max_negatives_per_reference=16)
    if frame.empty or frame["label"].nunique() < 2:
        raise ValueError("Training data needs both matched and unmatched candidate pairs.")
    groups = frame["reference_entity_id"].fillna(frame["source1_record_id"])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    fit_indices, validation_indices = next(splitter.split(frame, frame["label"], groups))
    fit = frame.iloc[fit_indices]
    validation = frame.iloc[validation_indices]
    model = RandomForestClassifier(
        n_estimators=160,
        max_depth=18,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(fit[FEATURE_COLUMNS], fit["label"])
    validation_candidates = validation[~validation["forced_positive"]].copy()
    probabilities = model.predict_proba(validation_candidates[FEATURE_COLUMNS])[:, 1]
    threshold = _threshold_at_precision(validation_candidates["label"].to_numpy(), probabilities, 0.99)
    total_validation_truth = int(
        truth[truth["source1_record_id"].isin(validation["source1_record_id"].unique())].shape[0]
    )
    validation_metrics = _evaluate_predictions(
        validation_candidates, probabilities, threshold, truth_count=total_validation_truth
    )
    output.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "training_rows": int(len(fit)),
        "training_positive_pairs": int(fit["label"].sum()),
        "training_negative_pairs": int((fit["label"] == 0).sum()),
        "candidate_count": int(len(frame)),
        "trained_at": pd.Timestamp.utcnow().isoformat(),
    }
    joblib.dump(bundle, output / MODEL_PATH_NAME, compress=3)
    return bundle


def load_model(model_dir: str | Path) -> dict[str, Any] | None:
    model_path = Path(model_dir) / MODEL_PATH_NAME
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def predict_for_reference(
    reference: pd.DataFrame,
    candidates: pd.DataFrame,
    source_channel: int,
    model_bundle: dict[str, Any],
    threshold: float | None = None,
    top_k: int = 16,
) -> pd.DataFrame:
    pairs = generate_candidate_pairs(reference, candidates, top_k=top_k)
    frame = candidate_feature_frame(reference, candidates, pairs, source_channel)
    if frame.empty:
        return pd.DataFrame(columns=[
            "source1_record_id", "matched_record_id", "match_probability", "business_name",
            "business_address", "country", "matched_source",
        ])
    probabilities = model_bundle["model"].predict_proba(frame[FEATURE_COLUMNS])[:, 1]
    frame["match_probability"] = probabilities
    frame["matched_source"] = "source2" if source_channel == 0 else "source3"
    frame = frame[frame["match_probability"] >= (threshold if threshold is not None else model_bundle["threshold"])]
    if frame.empty:
        return frame
    result = frame.merge(
        candidates[["record_id", "business_name", "business_address", "country"]],
        left_on="matched_record_id", right_on="record_id", how="left",
    )
    return result.drop(columns=["record_id"], errors="ignore").sort_values(
        "match_probability", ascending=False
    ).reset_index(drop=True)


def evaluate_test_set(data_dir: str | Path, model_bundle: dict[str, Any], threshold: float | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    references, source2, source3, truth = load_split(data_dir, "test")
    threshold = model_bundle["threshold"] if threshold is None else threshold
    results = []
    source_summaries: dict[str, Any] = {}
    for source, candidates, channel in (("source2", source2, 0), ("source3", source3, 1)):
        pairs = generate_candidate_pairs(references, candidates, top_k=12)
        positives = _truth_lookup(references, candidates, truth, source)
        frame = candidate_feature_frame(
            references, candidates, pairs, channel, positive_pairs=positives
        )
        source_truth_count = int((truth["matched_source"] == source).sum())
        probabilities = model_bundle["model"].predict_proba(frame[FEATURE_COLUMNS])[:, 1] if not frame.empty else np.array([])
        metrics = _evaluate_predictions(frame, probabilities, threshold, truth_count=source_truth_count)
        source_summaries[source] = metrics
        if not frame.empty:
            frame["match_probability"] = probabilities
            frame["matched_source"] = source
            results.append(frame)
    combined = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    total_truth = int(len(truth))
    combined_probabilities = combined["match_probability"].to_numpy() if not combined.empty else np.array([])
    combined_metrics = _evaluate_predictions(
        combined, combined_probabilities, threshold, truth_count=total_truth
    ) if not combined.empty else {
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "true_positives": 0,
        "false_positives": 0, "false_negatives": total_truth, "candidate_recall": 0.0,
        "predicted_matches": 0,
    }
    summary = {
        "threshold": threshold,
        "metrics": combined_metrics,
        "by_source": source_summaries,
        "source1_records": int(len(references)),
        "source2_records": int(len(source2)),
        "source3_records": int(len(source3)),
        "ground_truth_pairs": total_truth,
        "entity_count": int(references["entity_id"].nunique()),
        "no_match_entities": int(references["entity_id"].nunique() - truth["entity_id"].nunique()),
    }
    return summary, combined