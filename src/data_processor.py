import pandas as pd
import numpy as np
import os

def load_data(filepath):
    """Load data from CSV, handling encoding issues if any."""
    try:
        df = pd.read_csv(filepath, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """Clean the dataset."""
    # Remove rows with missing CustomerID
    df = df.dropna(subset=['CustomerID'])
    
    # Remove cancelled transactions (Quantity < 0)
    df = df[df['Quantity'] > 0]
    
    # Remove rows with invalid UnitPrice
    df = df[df['UnitPrice'] > 0]
    
    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Calculate TotalPrice
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    return df

def feature_engineering(df, reference_date=None):
    """RFM + extra features, computed only from transactions before reference_date."""
    if reference_date is None:
        reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    df = df[df['InvoiceDate'] < reference_date]

    rfm = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalPrice', 'sum'),
        Tenure=('InvoiceDate', lambda x: (reference_date - x.min()).days),
        Country=('Country', 'first'),
    ).reset_index()

    rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']
    diversity = df.groupby('CustomerID')['StockCode'].nunique().rename('UniqueProducts').reset_index()
    return pd.merge(rfm, diversity, on='CustomerID')


def label_churn(df, horizon_days=90):
    """Build a leak-free churn dataset.

    The previous version labelled churn as Recency > 90 and then used Recency
    as a feature, so every model scored ~100% by reading the label off an input.

    Now: pick a cutoff `horizon_days` before the end of the data. Features use
    only transactions before the cutoff; a customer is labelled churned (1) if
    they make no purchase in the `horizon_days` window after it.
    """
    cutoff = df['InvoiceDate'].max() + pd.Timedelta(days=1) - pd.Timedelta(days=horizon_days)
    features = feature_engineering(df, reference_date=cutoff)
    future_buyers = set(df.loc[df['InvoiceDate'] >= cutoff, 'CustomerID'])
    features['Churn'] = (~features['CustomerID'].isin(future_buyers)).astype(int)
    return features


def preprocess_pipeline(filepath):
    """Full pipeline."""
    print("Loading data...")
    df = load_data(filepath)
    
    print("Cleaning data...")
    df = clean_data(df)
    
    print("Building leak-free churn dataset...")
    return label_churn(df)

if __name__ == "__main__":
    # Path to data
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data.csv')
    
    if os.path.exists(data_path):
        print("Loading data...")
        raw_df = load_data(data_path)
        
        print("Cleaning data...")
        clean_df = clean_data(raw_df)
        
        # Save cleaned transaction data for Association Rules
        clean_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cleaned_transactions.csv')
        clean_df.to_csv(clean_output_path, index=False)
        print(f"Cleaned transactions saved to {clean_output_path}")

        print("Building leak-free churn dataset (features before cutoff, label after)...")
        rfm = label_churn(clean_df)

        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data.csv')
        rfm.to_csv(output_path, index=False)
        print(f"Processed RFM data saved to {output_path}")
        print(rfm.head())
        print(rfm['Churn'].value_counts(normalize=True))
    else:
        print(f"Data file not found at {data_path}")
