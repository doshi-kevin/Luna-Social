import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Configuration
VENUES_MASTER_PATH = "processed/venues_master.parquet"
POPULARITY_SCORES_PATH = "processed/popularity_scores.parquet"
ENGAGEMENT_CSV_PATH = "datasets/synthetic_engagement/engagement.csv"
OUTPUT_DIR = "processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "final_venue_features.parquet")

def create_output_directory():
    """Create processed output directory if it doesn't exist."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def load_all_datasets():
    """Load all three preprocessed datasets."""
    print("[*] Loading all preprocessed datasets...")
    
    # Load venues master
    print("   Loading venues_master.parquet...")
    venues_df = pd.read_parquet(VENUES_MASTER_PATH)
    print(f"   [+] Loaded {len(venues_df)} venues")
    
    # Load popularity scores
    print("   Loading popularity_scores.parquet...")
    popularity_df = pd.read_parquet(POPULARITY_SCORES_PATH)
    print(f"   [+] Loaded popularity scores for {len(popularity_df)} venues")
    
    # Load engagement data
    print("   Loading engagement.csv...")
    engagement_df = pd.read_csv(ENGAGEMENT_CSV_PATH)
    print(f"   [+] Loaded {len(engagement_df)} engagement records")
    print(f"   [+] Engagement columns: {list(engagement_df.columns)}")
    
    return venues_df, popularity_df, engagement_df

def process_engagement_data(engagement_df):
    """Process engagement data to compute user preference vectors."""
    print("[*] Processing engagement data...")
    
    # Expected columns: user_id, venue_id, seconds_viewed, clicked, category
    # Validate required columns
    required_cols = ['user_id', 'venue_id', 'seconds_viewed', 'clicked', 'category']
    missing_cols = [c for c in required_cols if c not in engagement_df.columns]
    
    if missing_cols:
        print(f"   [!] Warning: Missing columns {missing_cols}, adapting processing...")
        # If columns don't match exactly, try to infer
        if 'engagement_duration' in engagement_df.columns:
            engagement_df['seconds_viewed'] = engagement_df['engagement_duration']
        if 'is_clicked' in engagement_df.columns:
            engagement_df['clicked'] = engagement_df['is_clicked']
    
    # Ensure data types
    engagement_df['user_id'] = engagement_df['user_id'].astype(str)
    engagement_df['venue_id'] = engagement_df['venue_id'].astype(str)
    engagement_df['seconds_viewed'] = pd.to_numeric(engagement_df['seconds_viewed'], errors='coerce').fillna(0)
    engagement_df['clicked'] = engagement_df['clicked'].astype(int)
    
    print(f"   [+] Processed {len(engagement_df)} engagement records")
    
    return engagement_df

def compute_user_category_metrics(engagement_df):
    """Compute avg_seconds_viewed and click_rate per category per user."""
    print("[*] Computing user-category engagement metrics...")
    
    # Group by user and category
    user_category_metrics = engagement_df.groupby(['user_id', 'category']).agg({
        'seconds_viewed': 'mean',
        'clicked': lambda x: (x.sum() / len(x)) if len(x) > 0 else 0,
        'venue_id': 'count'
    }).reset_index()
    
    user_category_metrics.columns = [
        'user_id', 'category', 'avg_seconds_viewed', 'click_rate', 'interaction_count'
    ]
    
    print(f"   [+] Computed metrics for {len(user_category_metrics)} user-category pairs")
    print(f"   [+] Unique users: {user_category_metrics['user_id'].nunique()}")
    print(f"   [+] Unique categories: {user_category_metrics['category'].nunique()}")
    
    return user_category_metrics

def build_user_preference_vectors(user_category_metrics):
    """Build user preference vectors from category metrics."""
    print("[*] Building user preference vectors...")
    
    # Normalize metrics for each user across all categories
    scaler = StandardScaler()
    
    # Pivot to create user x category matrix
    pivot_seconds = user_category_metrics.pivot_table(
        index='user_id', 
        columns='category', 
        values='avg_seconds_viewed',
        fill_value=0
    )
    
    pivot_clicks = user_category_metrics.pivot_table(
        index='user_id',
        columns='category',
        values='click_rate',
        fill_value=0
    )
    
    # Combine metrics with weighted average
    preference_score = 0.6 * pivot_seconds + 0.4 * pivot_clicks
    
    # Normalize per user (sum to 1)
    preference_vectors = preference_score.div(preference_score.sum(axis=1), axis=0).fillna(0)
    
    print(f"   [+] Generated {len(preference_vectors)} user preference vectors")
    print(f"   [+] Preference vector shape: {preference_vectors.shape}")
    
    return preference_vectors, preference_score.columns.tolist()

def match_venues_to_categories(venues_df):
    """Create mapping of venues to their categories."""
    print("[*] Matching venues to categories...")

    venue_category_pairs = []

    for idx, row in venues_df.iterrows():
        business_id = row['business_id']
        raw = row['categories']

        # Normalize categories robustly
        if isinstance(raw, (list, tuple, set)):
            categories = list(raw)

        elif isinstance(raw, np.ndarray):  # <-- BUG FIX HERE
            categories = list(raw)

        elif raw is None:
            categories = []

        elif isinstance(raw, float) and pd.isna(raw):  # NaN
            categories = []

        else:
            # Treat anything else as comma-separated string
            categories = [c.strip() for c in str(raw).split(',') if c.strip()]

        # Store each venue-category pair
        for cat in categories:
            venue_category_pairs.append({
                'business_id': business_id,
                'category': cat,
                'venue_name': row['name'],
                'city': row['city'],
                'stars': row['stars'],
                'review_count': row['review_count']
            })

    venue_category_df = pd.DataFrame(venue_category_pairs)
    print(f"   [+] Created {len(venue_category_df)} venue-category pairs")

    return venue_category_df



def compute_category_popularity(popularity_df):
    """Compute popularity metrics aggregated at category (venue_name) level."""
    print("[*] Computing category-level popularity...")

    # In popularity_df, 'venue_name' actually holds the Foursquare venue category
    category_agg = popularity_df.groupby('venue_name').agg({
        'popularity_score': 'mean',
        'engagement_score': 'mean',
        'checkin_count': 'mean'
    }).reset_index()

    category_agg = category_agg.rename(columns={
        'venue_name': 'category',
        'popularity_score': 'category_avg_popularity',
        'engagement_score': 'category_avg_engagement',
        'checkin_count': 'category_avg_checkins'
    })

    print(f"   [+] Computed popularity for {len(category_agg)} categories")

    return category_agg


def build_final_feature_table(venues_df, venue_category_df, category_agg):
    """Build final ML-ready feature table by merging Yelp metadata with category-level stats."""
    print("[*] Building final feature table...")

    # Start from Yelp venues
    final_df = venues_df.copy()

    # Rename business_id to venue_id for consistency
    final_df = final_df.rename(columns={'business_id': 'venue_id'})

    # Join venue-category pairs with category-level popularity
    venue_cat_stats = venue_category_df.merge(
        category_agg,
        on='category',
        how='left'
    )

    # Aggregate category stats back to venue level
    venue_cat_agg = venue_cat_stats.groupby('business_id').agg({
        'category_avg_popularity': 'mean',
        'category_avg_engagement': 'mean',
        'category_avg_checkins': 'mean'
    }).reset_index()

    # Merge aggregated category stats into the venue table
    final_df = final_df.merge(
        venue_cat_agg,
        left_on='venue_id',
        right_on='business_id',
        how='left'
    ).drop(columns=['business_id'], errors='ignore')

    # ---- Feature engineering ----
    print("[*] Engineering additional features...")

    # Venue quality score (0–1)
    final_df['quality_score'] = (
        0.6 * (final_df['stars'] / 5.0) +
        0.2 * (final_df['review_count'] / final_df['review_count'].max()) +
        0.2 * final_df['category_avg_popularity'].fillna(0)
    )

    # Engagement potential: how likely people engage with this category
    final_df['engagement_potential'] = (
        final_df['category_avg_engagement'].fillna(0) *
        final_df['category_avg_popularity'].fillna(0)
    )

    max_potential = final_df['engagement_potential'].max()
    if max_potential > 0:
        final_df['engagement_potential'] = final_df['engagement_potential'] / max_potential

    # Fill NaNs in key numeric fields
    fill_columns = [
        'category_avg_popularity',
        'category_avg_engagement',
        'category_avg_checkins',
        'quality_score',
        'engagement_potential'
    ]
    for col in fill_columns:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna(0)

    # Select final feature columns
    feature_columns = [
        'venue_id', 'name', 'city', 'latitude', 'longitude',
        'stars', 'review_count', 'categories', 'total_checkins',
        'category_avg_popularity', 'category_avg_engagement', 'category_avg_checkins',
        'quality_score', 'engagement_potential'
    ]

    feature_columns = [c for c in feature_columns if c in final_df.columns]
    final_df = final_df[feature_columns].reset_index(drop=True)

    print(f"   [+] Final feature table shape: {final_df.shape}")
    print(f"   [+] Features: {list(final_df.columns)}")

    return final_df


def compute_feature_statistics(df):
    """Compute and display statistics on generated features."""
    print("[*] Feature statistics:")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        print(f"   {col}:")
        print(f"      Mean: {df[col].mean():.4f}")
        print(f"      Std:  {df[col].std():.4f}")
        print(f"      Min:  {df[col].min():.4f}")
        print(f"      Max:  {df[col].max():.4f}")
        print(f"      Nulls: {df[col].isnull().sum()}")

def save_final_features(df, output_path):
    """Save final feature table to parquet format."""
    print(f"[*] Saving final feature table to {output_path}...")
    
    table = pa.Table.from_pandas(df)

    pq.write_table(table, output_path, compression='snappy')
    
    print(f"[+] Saved {len(df)} venue records with engineered features")
    
    return df

def main():
    """Main execution pipeline."""
    print("="*60)
    print("FEATURE TABLE BUILDER - LUNA SOCIAL")
    print("="*60)
    
    try:
        create_output_directory()
        
        # Load all preprocessed datasets
        venues_df, popularity_df, engagement_df = load_all_datasets()
        
        # Process engagement data
        engagement_df = process_engagement_data(engagement_df)
        
        # Compute user-category metrics
        user_category_metrics = compute_user_category_metrics(engagement_df)
        
        # Build user preference vectors
        user_prefs, category_cols = build_user_preference_vectors(user_category_metrics)
        
        # Compute category popularity
                # Build venue-category mapping
        venue_category_df = match_venues_to_categories(venues_df)

        # Compute category popularity from Foursquare
        category_agg = compute_category_popularity(popularity_df)

        # Build final feature table
        final_df = build_final_feature_table(
            venues_df, venue_category_df, category_agg
        )

        
        # Compute feature statistics
        compute_feature_statistics(final_df)
        
        # Save final output
        save_final_features(final_df, OUTPUT_FILE)
        
        print("="*60)
        print(f"✓ COMPLETE: final_venue_features.parquet created")
        print(f"  Location: {OUTPUT_FILE}")
        print(f"  Records: {len(final_df)}")
        print(f"  Features: {len(final_df.columns)}")
        print("="*60)
        
    except Exception as e:
        print(f"[!] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()