# Business Entity Resolution

A local, end-to-end machine-learning demo that finds likely matches between three synthetic business registries. The demo has no external data or API dependencies.

## Run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

In this Replit project, the app is run by its configured workflow. On first start, the app creates the configured synthetic dataset if the TSVs are missing.

## Workflow

1. Explore the three noisy sources and their generated ground-truth pair labels.
2. Inspect normalization and candidate generation.
3. Train the local Random Forest model from the **Model Training** page.
4. Review precision, recall, F1, and candidate recall on the entity-disjoint test set.
5. Enter a new business on **Entity Matching** and inspect results in **Results**.
6. Download test metrics, predicted pairs, source TSVs, or query results.

## Dataset

`DATASET_SIZE = 30000` is the approximate total number of source records. The deterministic generator creates a train/test split by business entity, so an entity cannot occur in both splits. Each source TSV contains `record_id`, `entity_id`, `business_name`, `business_address`, and `country`. The `entity_id` column is a synthetic label and is never included in model features.

```text
data/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
    test_ground_truth.tsv
```

Regenerate the data with a stable seed:

```bash
python generate_data.py --size 30000 --seed 42
```

The generator includes case, punctuation, abbreviation, legal-suffix, spelling, whitespace, partial-address, and component-order variations, plus missing source records, duplicate records, and unrelated businesses with overlapping name tokens.

## Model

Country blocking and character n-gram nearest-neighbour retrieval create a manageable candidate set. A scikit-learn Random Forest scores each pair using name/address character and token similarities, exact/containment checks, length ratios, house-number agreement, country, and source channel. A held-out group of training entities calibrates a precision-oriented confidence threshold. The test metrics are pair-level metrics over generated candidates, with candidates missed by blocking counted as false negatives.