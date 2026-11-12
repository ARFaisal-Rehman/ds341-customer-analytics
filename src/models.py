import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'src', 'processed_data.csv')
TRANS_PATH = os.path.join(BASE_DIR, 'src', 'cleaned_transactions.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def train_eval_models(df):
    """Train and evaluate classification models."""
    print("Training Classification Models...")
    
    # Preprocessing
    X = df.drop(['CustomerID', 'Churn', 'Country'], axis=1) # Drop non-numeric/ID
    y = df['Churn']
    
    # Encode Country if needed (skipping for simplicity or using frequency encoding could be better, but let's drop for now)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'ROC_AUC': roc_auc_score(y_test, y_prob) if y_prob is not None else None
        }
        
        results[name] = metrics
        
        # Save confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(RESULTS_DIR, f'cm_{name.replace(" ", "_")}.png'))
        plt.close()
        
    # Save results to JSON
    with open(os.path.join(RESULTS_DIR, 'model_metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    return results

def perform_clustering(df):
    """Perform customer segmentation using K-Means."""
    print("Performing Clustering...")
    features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue']
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Elbow method (simplified: just picking k=3 or 4)
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    df['Cluster'] = clusters
    
    # Save cluster summary
    cluster_summary = df.groupby('Cluster')[features].mean().reset_index()
    cluster_summary.to_json(os.path.join(RESULTS_DIR, 'cluster_summary.json'), orient='records')
    
    # Visualize Clusters (PCA)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis')
    plt.title('Customer Segments (PCA)')
    plt.savefig(os.path.join(RESULTS_DIR, 'clusters_pca.png'))
    plt.close()
    
    return df

def mine_association_rules():
    """Mine association rules using Apriori."""
    print("Mining Association Rules...")
    try:
        df = pd.read_csv(TRANS_PATH)
        # Filter for top products to save memory/time
        top_products = df['Description'].value_counts().head(50).index
        df = df[df['Description'].isin(top_products)]
        
        basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
                  .sum().unstack().reset_index().fillna(0)
                  .set_index('InvoiceNo'))
        
        def encode_units(x):
            if x <= 0: return 0
            if x >= 1: return 1
            
        basket_sets = basket.map(encode_units)
        
        frequent_itemsets = apriori(basket_sets, min_support=0.03, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        
        # Save top rules
        top_rules = rules.sort_values('lift', ascending=False).head(10)
        top_rules['antecedents'] = top_rules['antecedents'].apply(lambda x: list(x))
        top_rules['consequents'] = top_rules['consequents'].apply(lambda x: list(x))
        
        top_rules.to_json(os.path.join(RESULTS_DIR, 'association_rules.json'), orient='records')
    except Exception as e:
        print(f"Association Rule Mining failed: {e}")

if __name__ == "__main__":
    df = load_data()
    
    # 1. Classification
    train_eval_models(df)
    
    # 2. Clustering
    df_clustered = perform_clustering(df)
    
    # 3. Association Rules
    mine_association_rules()
    
    print("Modeling pipeline completed.")
