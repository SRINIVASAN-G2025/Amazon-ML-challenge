"""Deterministic synthetic business registry generation."""

from __future__ import annotations

import random
import re
import string
from pathlib import Path
from typing import Any

import pandas as pd

DATASET_SIZE = 30_000
SEED = 42

BUSINESS_STARTS = [
    "Sri Lakshmi", "Northstar", "Apex", "Blue River", "Golden Gate",
    "Evergreen", "Cedar Ridge", "Silverline", "Orchid", "Meridian",
    "Pioneer", "Summit", "Crescent", "Atlas", "Harbor", "Maple Leaf",
    "Brightpath", "Redwood", "Cobalt", "Vantage", "Sunrise", "Trident",
    "Oak & Stone", "Everwell", "Sterling", "Meadowbrook", "Noble",
    "Clearwater", "Indigo", "Westbridge", "Lighthouse", "Monarch",
]
INDUSTRIES = [
    "Textiles", "Engineering", "Foods", "Logistics", "Pharmaceuticals",
    "Packaging", "Electricals", "Trading", "Consulting", "Technologies",
    "Automotive", "Agriculture", "Furniture", "Chemicals", "Hospitality",
    "Healthcare", "Construction", "Paper Mills", "Import Export",
    "Renewable Energy", "Wholesale", "Digital Systems", "Ceramics",
    "Precision Tools", "Financial Services",
]
LEGAL_SUFFIXES = [
    "Private Limited", "Pvt Ltd", "Limited", "LLC", "Inc.", "SAS",
    "SARL", "S.A.", "GmbH", "Corporation", "and Sons", "",
]

COUNTRY_LOCATIONS: dict[str, dict[str, list[str]]] = {
    "India": {
        "cities": ["Coimbatore", "Pune", "Jaipur", "Ahmedabad", "Indore",
                   "Chennai", "Hyderabad", "Surat", "Kochi", "Nagpur"],
        "streets": ["Gandhi Road", "Nehru Street", "MG Road", "Lake View Road",
                    "Market Road", "Station Road", "Industrial Estate Road",
                    "Park Avenue", "Temple Street", "Ring Road"],
        "areas": ["Peelamedu", "Civil Lines", "Gandhipuram", "Anna Nagar",
                  "Industrial Area", "Sector 4", "Old Town", "Central Market"],
    },
    "USA": {
        "cities": ["Austin", "Denver", "Columbus", "Portland", "Raleigh",
                   "Phoenix", "Madison", "Tampa", "Richmond", "Boise"],
        "streets": ["Market Street", "Oak Avenue", "Washington Boulevard",
                    "Lake Shore Drive", "Pine Street", "Commerce Road",
                    "Broadway", "Cedar Lane", "Union Street", "Main Street"],
        "areas": ["North Park", "Downtown", "Riverside", "West End",
                  "Civic Center", "East Side", "Warehouse District", "Midtown"],
    },
    "France": {
        "cities": ["Lyon", "Nantes", "Toulouse", "Lille", "Bordeaux",
                   "Marseille", "Grenoble", "Dijon", "Rennes", "Tours"],
        "streets": ["Rue de la Paix", "Avenue Victor Hugo", "Rue Nationale",
                    "Boulevard Gambetta", "Rue des Lilas", "Place du Marché",
                    "Avenue de Lyon", "Rue du Port", "Rue Pasteur", "Rue Centrale"],
        "areas": ["Centre Ville", "Les Quais", "Quartier Nord", "Saint Michel",
                  "Zone Artisanale", "Parc Central", "Vieux Port", "La Gare"],
    },
}


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 5:
        return word
    position = rng.randrange(1, len(word) - 1)
    operation = rng.choice(("swap", "drop", "repeat"))
    if operation == "swap":
        chars = list(word)
        chars[position], chars[position + 1] = chars[position + 1], chars[position]
        return "".join(chars)
    if operation == "drop":
        return word[:position] + word[position + 1 :]
    return word[:position] + word[position] + word[position:]


def _spell_business_name(name: str, rng: random.Random) -> str:
    words = name.split()
    if not words:
        return name
    suffix = words[-1].lower().rstrip(".,")
    if suffix in {"limited", "ltd", "inc", "incorporated", "llc", "sas", "sarl"}:
        if rng.random() < 0.62:
            words.pop()
        if words and words[-1].lower().rstrip(".,") in {"private", "pvt"} and rng.random() < 0.55:
            words.pop()
    if rng.random() < 0.2:
        index = rng.randrange(len(words))
        words[index] = _typo(words[index], rng)
    if rng.random() < 0.08 and len(words) > 2:
        words = words[1:] + words[:1]
    return " ".join(words)


def _noisy_name(name: str, rng: random.Random) -> str:
    value = _spell_business_name(name, rng)
    if rng.random() < 0.32:
        value = value.upper() if rng.random() < 0.5 else value.lower()
    if rng.random() < 0.3:
        value = re.sub(r"\bPrivate Limited\b", "Pvt. Ltd.", value, flags=re.I)
        value = re.sub(r"\bPvt Ltd\b", "Private Limited", value, flags=re.I)
        value = re.sub(r"\bLimited\b", "Ltd.", value, flags=re.I)
    if rng.random() < 0.18:
        value = re.sub(r"\s+", "   ", value.strip())
    if rng.random() < 0.2:
        value = value.replace(" ", rng.choice((" & ", "-", ", ")), 1)
    return value.strip()


def _noisy_address(address: str, rng: random.Random) -> str:
    parts = [part.strip() for part in address.split(",")]
    value = address
    if rng.random() < 0.38:
        value = re.sub(r"\bRoad\b", "Rd", value, flags=re.I)
        value = re.sub(r"\bStreet\b", "St", value, flags=re.I)
        value = re.sub(r"\bBoulevard\b", "Blvd", value, flags=re.I)
        value = re.sub(r"\bAvenue\b", "Ave", value, flags=re.I)
    if rng.random() < 0.18:
        value = re.sub(r"\b(\d+)\s+", r"\1, ", value, count=1)
    if rng.random() < 0.2 and len(parts) > 2:
        value = ", ".join([parts[-1], parts[0], *parts[1:-1]])
    if rng.random() < 0.22 and len(parts) > 2:
        value = ", ".join(parts[:2])
    if rng.random() < 0.12:
        value = re.sub(r"\b([A-Za-z]{5,})\b", lambda m: _typo(m.group(1), rng), value, count=1)
    if rng.random() < 0.24:
        value = value.upper() if rng.random() < 0.5 else value.lower()
    if rng.random() < 0.2:
        value = value.replace(",", rng.choice(("", ".", " ,")))
    if rng.random() < 0.12:
        value = "  " + value + "   "
    return value.strip()


def _make_business(rng: random.Random, index: int) -> dict[str, str]:
    country = rng.choice(tuple(COUNTRY_LOCATIONS))
    location = COUNTRY_LOCATIONS[country]
    city = rng.choice(location["cities"])
    street = rng.choice(location["streets"])
    area = rng.choice(location["areas"])
    start = rng.choice(BUSINESS_STARTS)
    industry = rng.choice(INDUSTRIES)
    suffix = rng.choice(LEGAL_SUFFIXES)
    base_name = " ".join(part for part in (start, industry, suffix) if part)
    # A minority of names get a modest local identifier so unrelated businesses
    # can share common words without making the synthetic registry collision-prone.
    if rng.random() < 0.36:
        base_name = f"{base_name} {rng.choice(('Group', 'Works', 'Partners', 'No.'))} {index % 997 + 1:03d}"
    number = rng.randint(1, 998)
    unit = rng.choice(("", "", "", f", Suite {rng.randint(2, 240)}"))
    address = f"{number} {street}{unit}, {area}, {city}"
    return {
        "entity_id": f"BIZ-{index:06d}",
        "business_name": base_name,
        "business_address": address,
        "country": country,
    }


def _make_record(entity: dict[str, str], source: str, record_no: int, rng: random.Random) -> dict[str, Any]:
    if source == "source1":
        name, address = entity["business_name"], entity["business_address"]
    else:
        name = _noisy_name(entity["business_name"], rng)
        address = _noisy_address(entity["business_address"], rng)
    return {
        "record_id": f"{source.upper()}-{record_no:07d}",
        **entity,
        "business_name": name,
        "business_address": address,
    }


def _write_split(entities: list[dict[str, str]], split: str, out_dir: Path, rng: random.Random) -> int:
    sources: dict[str, list[dict[str, Any]]] = {name: [] for name in ("source1", "source2", "source3")}
    truth: list[dict[str, str]] = []
    counters = {name: 1 for name in sources}

    for entity in entities:
        ref = _make_record(entity, "source1", counters["source1"], rng)
        counters["source1"] += 1
        sources["source1"].append(ref)
        for source in ("source2", "source3"):
            absent_chance = 0.085 if source == "source2" else 0.105
            if rng.random() < absent_chance:
                continue
            count = 2 if rng.random() < 0.115 else 1
            for _ in range(count):
                record = _make_record(entity, source, counters[source], rng)
                counters[source] += 1
                sources[source].append(record)
                truth.append({
                    "entity_id": entity["entity_id"],
                    "matched_source": source,
                    "source1_record_id": ref["record_id"],
                    "matched_record_id": record["record_id"],
                })

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "train" if split == "train" else "test"
    for source, records in sources.items():
        frame = pd.DataFrame(records, columns=[
            "record_id", "entity_id", "business_name", "business_address", "country"
        ])
        frame.to_csv(out_dir / f"{prefix}_{source}.tsv", sep="\t", index=False)
    pd.DataFrame(truth, columns=[
        "entity_id", "matched_source", "source1_record_id", "matched_record_id"
    ]).to_csv(out_dir / f"{prefix}_ground_truth.tsv", sep="\t", index=False)
    return sum(len(rows) for rows in sources.values())


def generate_dataset(
    root: str | Path,
    dataset_size: int = DATASET_SIZE,
    seed: int = SEED,
) -> dict[str, int]:
    """Write a reproducible entity-disjoint train/test set near the record target."""
    if dataset_size < 300:
        raise ValueError("dataset_size must be at least 300 records")
    rng = random.Random(seed)
    entity_count = max(100, round(dataset_size / 3.05))
    entities = [_make_business(rng, index + 1) for index in range(entity_count)]
    rng.shuffle(entities)
    split_at = int(len(entities) * 0.8)
    root = Path(root)
    train_records = _write_split(entities[:split_at], "train", root / "train", rng)
    test_records = _write_split(entities[split_at:], "test", root / "test", rng)
    return {
        "business_entities": entity_count,
        "target_records": dataset_size,
        "train_records": train_records,
        "test_records": test_records,
        "total_records": train_records + test_records,
    }