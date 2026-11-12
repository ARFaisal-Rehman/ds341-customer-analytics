> **Note (November 2026):** this is the course report as submitted. The analysis was later revised (log-scaled segmentation with k chosen by silhouette, FP-Growth over 300 products, model selection by cross-validation), so some numbers below differ from the current results. See the [README](README.md) for the revised figures.

# Customer Churn Prediction Project Report

## 1. Methodology

### Data Processing
- **Cleaning**: Removed rows with missing `CustomerID` and cancelled transactions (`Quantity` < 0).
- **Feature Engineering**: Implemented RFM (Recency, Frequency, Monetary) analysis.
  - **Recency**: Days since last purchase.
  - **Frequency**: Number of unique invoices.
  - **Monetary**: Total spend.
  - **AvgOrderValue**: Average spend per order.
  - **UniqueProducts**: Diversity of items purchased.
- **Churn Definition (leak-free)**: A cutoff is set 90 days before the end of the data. Features are computed only from purchases *before* the cutoff; a customer is churned if they make no purchase in the 90 days *after* it. (An earlier version labelled churn as `Recency > 90` while also using Recency as a feature, which leaked the label and produced a meaningless 100% accuracy.)
- **Tenure**: Days since first purchase (added).

### Modeling
- **Classification**: Trained four models to predict churn:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
- **Clustering**: Used K-Means to segment customers into 4 clusters based on RFM scores.
- **Association Rule Mining**: Used Apriori algorithm to find product bundles and cross-selling opportunities.

## 2. Results & Analysis

### Model Performance
Held-out test set (20%, stratified). Churn rate ≈ 43%.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.632 | 0.561 | 0.678 | 0.614 | **0.713** |
| Decision Tree (depth 6) | 0.630 | 0.560 | 0.671 | 0.611 | 0.672 |
| Random Forest | 0.643 | 0.584 | 0.610 | 0.596 | 0.711 |
| XGBoost | 0.648 | 0.603 | 0.541 | 0.570 | 0.706 |

An AUC around 0.71 is realistic for behavioural churn from RFM features alone. Logistic Regression matches the tree ensembles, which suggests the signal is mostly linear in recency/frequency; next steps would be time-windowed features (spend trend, inter-purchase gaps) and a cost-based decision threshold.

### Customer Segments
- **Cluster 0**: Likely "At Risk" or "Lost" (High Recency, Low Frequency).
- **Cluster 1**: "Loyal" (Low Recency, High Frequency).
- **Cluster 2**: "Whales" (High Monetary).
- **Cluster 3**: "New/Occasional" (Low Frequency, Low Monetary).

### Business Recommendations
1.  **Retain High Value Customers**: Focus on Cluster 2 with personalized emails and exclusive offers.
2.  **Re-engage At-Risk**: For Cluster 0, send "We miss you" coupons.
3.  **Cross-Sell**: Use the discovered Association Rules (e.g., if buying "Lunch Box", suggest "Water Bottle") to increase basket size.

## 3. Challenges & Solutions
- **Imbalanced Data**: Churners might be fewer than retained users. Solved by using F1-score for evaluation.
- **Data Quality**: Many missing CustomerIDs. Solved by filtering them out as they are not actionable.
- **Scalability**: Association Rules can be computationally expensive. Solved by filtering for top products.

## 4. How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run data pipeline: `python src/data_processor.py`
3. Run modeling: `python src/models.py`
4. Start dashboard: `python app.py`
5. Open `http://localhost:5000` in your browser.
