import json
import tarfile
import pandas as pd
import pyarrow as pa
import numpy as np
import pyarrow.parquet as pq
import os
from pathlib import Path

YELP_TAR_PATH = "datasets/yelp/yelp_dataset.tar"
OUTPUT_DIR = "processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "venues_master.parquet")


def create_output_directory():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# CATEGORY HELPERS
# ---------------------------------------------------------

def normalize_categories(df):
    print("[*] Normalizing categories...")

    def parse(cat):
        if pd.isna(cat):
            return []
        if isinstance(cat, list):
            return [str(c).strip() for c in cat if c]
        if isinstance(cat, str):
            return [c.strip() for c in cat.split(",") if c.strip()]
        return []

    df["categories"] = df["categories"].apply(parse)
    print("[+] Categories normalized")
    return df


def fix_category_list(cat):
    if isinstance(cat, list):
        return [str(c) for c in cat]
    if isinstance(cat, np.ndarray):
        return [str(c) for c in cat.tolist()]
    if isinstance(cat, str):
        return [c.strip() for c in cat.split(",") if c.strip()]
    return []


# ---------------------------------------------------------
# EXTRACTION + LOADING
# ---------------------------------------------------------

def extract_yelp_tar():
    print("[*] Extracting Yelp dataset...")
    extract_path = "yelp_extracted"

    with tarfile.open(YELP_TAR_PATH, "r") as tar:
        tar.extractall(path=extract_path)

    business = None
    checkin = None

    for root, dirs, files in os.walk(extract_path):
        for f in files:
            lf = f.lower()
            if "business" in lf and lf.endswith(".json"):
                business = os.path.join(root, f)
            if "checkin" in lf and lf.endswith(".json"):
                checkin = os.path.join(root, f)

    if not business or not checkin:
        raise FileNotFoundError("business.json / checkin.json NOT found")

    print(f"[+] business.json → {business}")
    print(f"[+] checkin.json → {checkin}")

    return business, checkin


def load_business_data(path):
    print("[*] Loading business.json...")

    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except:
                pass

    df = pd.DataFrame(items)

    print(f"[+] Loaded {len(df)} business entries")

    cols = ["business_id", "name", "city", "latitude", "longitude",
            "stars", "review_count", "categories"]

    return df[cols].copy()


def clean_venues_data(df):
    print("[*] Cleaning venues...")

    df = df.drop_duplicates(subset=["business_id"])
    df = df.dropna(subset=["business_id", "name", "latitude", "longitude"])

    df["business_id"] = df["business_id"].astype(str)
    df["name"] = df["name"].astype(str)
    df["city"] = df["city"].astype(str)

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0).astype(int)

    df = df.dropna(subset=["latitude", "longitude"])

    print(f"[+] Cleaned venues → {len(df)} rows")
    return df


def load_checkins(path):
    print("[*] Loading checkins...")

    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                out[obj["business_id"]] = len(obj.get("time", []))
            except:
                pass

    print(f"[+] Loaded {len(out)} checkin rows")
    return out


def merge_with_checkins(df, checkins):
    df["total_checkins"] = df["business_id"].map(checkins).fillna(0).astype(int)
    return df


# ---------------------------------------------------------
# MANUAL PARQUET SAVER  (THE REAL FIX)
# ---------------------------------------------------------

def save_venues_master(df, out_path):
    print("[*] Saving parquet with manual Arrow schema...")

    df["categories"] = df["categories"].apply(fix_category_list)

    table = pa.Table.from_arrays(
        [
            pa.array(df["business_id"].astype(str).tolist(), pa.string()),
            pa.array(df["name"].astype(str).tolist(), pa.string()),
            pa.array(df["city"].astype(str).tolist(), pa.string()),
            pa.array(df["latitude"].tolist(), pa.float64()),
            pa.array(df["longitude"].tolist(), pa.float64()),
            pa.array(df["stars"].tolist(), pa.float64()),
            pa.array(df["review_count"].tolist(), pa.int64()),
            pa.array(df["categories"].tolist(), pa.list_(pa.string())),
            pa.array(df["total_checkins"].tolist(), pa.int64()),
        ],
        names=[
            "business_id", "name", "city", "latitude", "longitude",
            "stars", "review_count", "categories", "total_checkins"
        ]
    )

    pq.write_table(table, out_path, compression="snappy")
    print("[✓] Parquet saved successfully")


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

def main():
    print("=" * 60)
    print("YELP PREPROCESSING PIPELINE")
    print("=" * 60)

    create_output_directory()

    business_path, checkin_path = extract_yelp_tar()

    df = load_business_data(business_path)
    df = normalize_categories(df)

    # 🔥 CRITICAL FIX: force real Python lists BEFORE cleaning
    df["categories"] = df["categories"].apply(
        lambda x: [str(i) for i in x] if isinstance(x, list) else []
    )

    df = clean_venues_data(df)

    checkins = load_checkins(checkin_path)
    df = merge_with_checkins(df, checkins)

    save_venues_master(df, OUTPUT_FILE)

    print("=" * 60)
    print("✓ DONE — venues_master.parquet created")
    print(f"Location: {OUTPUT_FILE}")
    print(f"Records: {len(df)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
