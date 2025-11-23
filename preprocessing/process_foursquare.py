import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
from pathlib import Path
from datetime import datetime

# Configuration
FOURSQUARE_CSV_PATH = "datasets/foursquare/dataset_TSMC2014_NYC.csv"
OUTPUT_DIR = "processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "popularity_scores.parquet")

def create_output_directory():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def load_foursquare_data():
    print("[*] Loading Foursquare dataset...")
    df = pd.read_csv(FOURSQUARE_CSV_PATH)
    print(f"[+] Loaded {len(df)} records")
    print(f"[+] Columns: {list(df.columns)}")
    return df

def parse_timestamps(df):
    print("[*] Parsing timestamps...")

    # Your dataset uses human-readable timestamps like:
    # Tue Apr 03 18:00:09 +0000 2012
    df['utc_timestamp'] = pd.to_datetime(
        df['utc_timestamp'],
        format='%a %b %d %H:%M:%S %z %Y',
        errors='coerce'
    )

    if df['utc_timestamp'].isna().any():
        print("[!] Some timestamps failed to parse:")
        print(df[df['utc_timestamp'].isna()].head())

    df['hour'] = df['utc_timestamp'].dt.hour
    df['date'] = df['utc_timestamp'].dt.date

    print(f"[+] Parsed timestamps: {df['utc_timestamp'].min()} → {df['utc_timestamp'].max()}")
    return df

def aggregate_venue_metrics(df):
    print("[*] Aggregating venue metrics...")

    venue_groups = df.groupby('venue_id').agg({
        'user_id': lambda x: x.nunique(),
        'utc_timestamp': 'count',
        'venue_name': 'first'
    }).reset_index()

    venue_groups.columns = [
        'venue_id',
        'distinct_user_count',
        'checkin_count',
        'venue_name'
    ]

    print(f"[+] Aggregated {len(venue_groups)} venues")
    return df, venue_groups

def compute_hourly_distribution(df):
    print("[*] Computing hourly distributions...")

    hourly_dist = df.groupby(['venue_id', 'hour']).size().unstack(fill_value=0)

    for hour in range(24):
        if hour not in hourly_dist.columns:
            hourly_dist[hour] = 0

    hourly_dist = hourly_dist[sorted(hourly_dist.columns)]
    hourly_dist_norm = hourly_dist.div(hourly_dist.sum(axis=1), axis=0).fillna(0)

    hourly_vectors = {
        venue_id: row.tolist() for venue_id, row in hourly_dist_norm.iterrows()
    }

    print(f"[+] Generated hourly vectors for {len(hourly_vectors)} venues")
    return hourly_vectors

def build_popularity_scores(venue_groups, hourly_vectors):
    print("[*] Building popularity scores...")

    venue_groups['checkin_score'] = \
        venue_groups['checkin_count'] / venue_groups['checkin_count'].max()

    venue_groups['user_diversity_score'] = \
        venue_groups['distinct_user_count'] / venue_groups['distinct_user_count'].max()

    venue_groups['engagement_score'] = (
        venue_groups['checkin_count'] /
        venue_groups['distinct_user_count']
    ).fillna(0)

    if venue_groups['engagement_score'].max() > 0:
        venue_groups['engagement_score'] /= venue_groups['engagement_score'].max()

    venue_groups['popularity_score'] = (
        0.4 * venue_groups['checkin_score'] +
        0.3 * venue_groups['user_diversity_score'] +
        0.3 * venue_groups['engagement_score']
    )

    venue_groups['hourly_distribution'] = \
        venue_groups['venue_id'].map(hourly_vectors)

    final_cols = [
        'venue_id', 'venue_name', 'checkin_count', 'distinct_user_count',
        'checkin_score', 'user_diversity_score', 'engagement_score',
        'popularity_score', 'hourly_distribution'
    ]

    venue_groups = venue_groups[final_cols]
    print("[+] Popularity scores computed")
    return venue_groups

def save_popularity_scores(df, output_path):
    print(f"[*] Saving popularity scores → {output_path}")
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path, compression='snappy')
    print("[+] Saved!")

def main():
    print("="*60)
    print("FOURSQUARE DATA PREPROCESSING PIPELINE")
    print("="*60)

    df = load_foursquare_data()

    # Standardize column names
    df = df.rename(columns={
        'userId': 'user_id',
        'venueId': 'venue_id',
        'utcTimestamp': 'utc_timestamp',
        'venueCategory': 'venue_name'
    })

    df = parse_timestamps(df)

    df, venue_groups = aggregate_venue_metrics(df)

    hourly_vectors = compute_hourly_distribution(df)

    venue_groups = build_popularity_scores(venue_groups, hourly_vectors)

    save_popularity_scores(venue_groups, OUTPUT_FILE)

    print("="*60)
    print(f"✓ DONE — Saved {len(venue_groups)} popularity scores")
    print("="*60)

if __name__ == "__main__":
    main()
